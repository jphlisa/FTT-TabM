import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# =========================
# 0) Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) FT-Transformer Backbone (minimal & stable)
# =========================
class GEGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_ff),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformerBackbone(nn.Module):
    """
    Tokenize each feature into a token embedding of size d_model.
    - Numeric: per-feature affine -> token
    - Categorical: embedding table -> token
    Add [CLS] token, transformer blocks, output CLS vector.
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 384,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if n_num_features > 0:
            self.num_weight = nn.Parameter(torch.randn(n_num_features, d_model) * 0.01)
            self.num_bias = nn.Parameter(torch.zeros(n_num_features, d_model))
        else:
            self.num_weight, self.num_bias = None, None

        self.cat_embeds = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.cls_token)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        B = x_num.size(0) if x_num.numel() > 0 else x_cat.size(0)
        tokens = [self.cls_token.expand(B, 1, self.d_model)]

        if self.n_num_features > 0:
            xn = x_num.unsqueeze(-1)  # [B, N, 1]
            num_tok = xn * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)  # [B, N, D]
            tokens.append(num_tok)

        if len(self.cat_cardinalities) > 0:
            cat_toks = []
            for j, emb in enumerate(self.cat_embeds):
                cj = x_cat[:, j]  # [B]
                cat_toks.append(emb(cj).unsqueeze(1))  # [B, 1, D]
            tokens.append(torch.cat(cat_toks, dim=1))  # [B, N_cat, D]

        x = torch.cat(tokens, dim=1)  # [B, T, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        return x[:, 0, :]  # CLS vector


# =========================
# 2) TabM Head (Parameter-Efficient Ensemble)
# =========================
class TabMHead(nn.Module):
    """K linear heads; average logits."""
    def __init__(self, d_in: int, n_classes: int, K: int = 8):
        super().__init__()
        self.K = K
        self.W = nn.Parameter(torch.randn(K, d_in, n_classes) * 0.02)
        self.b = nn.Parameter(torch.zeros(K, n_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits_k = torch.einsum("bd,kdc->bkc", h, self.W) + self.b.unsqueeze(0)
        return logits_k.mean(dim=1)


class FTTransformerTabM(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list,
        n_classes: int = 2,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 384,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        K: int = 8,
    ):
        super().__init__()
        self.backbone = FTTransformerBackbone(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.head = TabMHead(d_in=d_model, n_classes=n_classes, K=K)

    def forward(self, x_num, x_cat):
        h = self.backbone(x_num, x_cat)
        return self.head(h)


# =========================
# 3) Training / Evaluation
# =========================
def train(model, Xn_train, Xc_train, y_train, Xn_val, Xc_val, y_val, device, epochs=200, lr=2e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    Xn_train = torch.tensor(Xn_train, dtype=torch.float32, device=device)
    Xc_train = torch.tensor(Xc_train, dtype=torch.long, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)

    Xn_val = torch.tensor(Xn_val, dtype=torch.float32, device=device)
    Xc_val = torch.tensor(Xc_val, dtype=torch.long, device=device)
    y_val = torch.tensor(y_val, dtype=torch.long, device=device)

    best_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xn_train, Xc_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xn_val, Xc_val)
                val_pred = val_logits.argmax(dim=1).detach().cpu().numpy()
                val_true = y_val.detach().cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, zero_division=0)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(f"Epoch {ep:03d} | loss={loss.item():.4f} | val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_all(model, Xn_test, Xc_test, y_test, device):
    model.eval()
    Xn_test = torch.tensor(Xn_test, dtype=torch.float32, device=device)
    Xc_test = torch.tensor(Xc_test, dtype=torch.long, device=device)

    logits = model(Xn_test, Xc_test)
    pred = logits.argmax(dim=1).detach().cpu().numpy()

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    cm = confusion_matrix(y_test, pred)
    report = classification_report(
        y_test, pred,
        target_names=["Fail (G3<10)", "Pass (G3>=10)"],
        digits=4,
        zero_division=0
    )

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm, "report": report}


# =========================
# 4) Main
# =========================
def main():
    set_seed(36)
    CSV_PATH = "E:/project/FT-TabM/datasets/UCI/student-mat.csv"  # 参考知识 Change it to your path as well

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cannot find {CSV_PATH}. Put it in the same folder or change CSV_PATH.")

    df = pd.read_csv(CSV_PATH, sep=";")

    # label: Pass/Fail by G3
    y = (df["G3"] >= 10).astype(int).values

    # numeric & categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove("G3")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # preprocess numeric
    scaler = StandardScaler()
    Xn = scaler.fit_transform(df[num_cols].to_numpy(dtype=np.float32))

    # preprocess categorical
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xc = enc.fit_transform(df[cat_cols].astype(str)).astype(np.int64)
    Xc = Xc + 1  # unknown(-1)->0, known start at 1
    cat_cardinalities = [len(cats) + 1 for cats in enc.categories_]  # +1 for unknown token

    # split: train/val/test = 70/15/15 (stratified)
    Xn_train, Xn_temp, Xc_train, Xc_temp, y_train, y_temp = train_test_split(
        Xn, Xc, y, test_size=0.30, random_state=36, stratify=y
    )
    Xn_val, Xn_test, Xc_val, Xc_test, y_val, y_test = train_test_split(
        Xn_temp, Xc_temp, y_temp, test_size=0.50, random_state=36, stratify=y_temp
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FTTransformerTabM(
        n_num_features=Xn.shape[1],
        cat_cardinalities=cat_cardinalities,
        n_classes=2,
        d_model=192,
        n_layers=3,
        n_heads=8,
        d_ff=384,
        dropout=0.1,
        attn_dropout=0.1,
        K=8,
    )

    model = train(
        model,
        Xn_train, Xc_train, y_train,
        Xn_val, Xc_val, y_val,
        device=device,
        epochs=200,
        lr=2e-3,
    )

    results = evaluate_all(model, Xn_test, Xc_test, y_test, device)

    print("\n==== Test Metrics (Pass/Fail) ====")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-score : {results['f1']:.4f}")

    print("\n==== Confusion Matrix (rows=true, cols=pred) ====")
    print(results["cm"])

    print("\n==== Classification Report ====")
    print(results["report"])


if __name__ == "__main__":
    main()
