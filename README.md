# FTT-TabM

 Student performance prediction is an important task in Educational Data Mining (EDM), aiming to support timely interventions and personalized learning by analyzing students’ behavioral and academic data. However, modeling complex feature interactions in tabular educational data remains challenging for conventional machine learning methods. To address this issue, this paper proposes a student performance prediction model, termed FTT-TabM, which combines an FT-Transformer backbone with a parameter-sharing multi-head prediction mechanism inspired by TabM. The FT-Transformer is employed to capture high-order feature interactions through self-attention, while the multi-head prediction structure aims to improve model robustness by learning diverse decision boundaries with shared representations. Experiments are conducted on four real-world educational datasets, including UCI-Mat, UCI-Por, OULAD, and AiEducation. The results show that the proposed method achieves competitive and, in most cases, improved performance compared with several classical machine learning baselines. In addition, ablation studies are performed to analyze the contributions of the Transformer backbone and the multi-head prediction mechanism. Overall, the proposed approach provides a practical solution for modeling tabular educational data and offers potential value for student performance prediction tasks.

---

## 📁 Project Structure

```
TabPFN-SPP 
├── main.py                        ← Experimental
├──datasets/  
│   ├── student-mat.csv    
│   └── student-por.csv  
├──output/   
│  
├── requirements.txt               ← Python dependencies  
└── README.md                      ← Project overview
```

---

##  Environment Setup

- **Python version**: `3.11.4`  
- **Framework**: `Torch`  
- **Recommended**: Use Anaconda

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

##  Dataset & Results

All datasets and result files are publicly available https://archive.ics.uci.edu/dataset/320/student+performance
This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por).Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful.-



---

##  Running Experiments

The experiment is run in main. py

Example:

```bash
python main.py
```

If you encounter errors like `ModuleNotFound`, copy files from `Utils/` into the current folder or adjust Python import paths.

---

##  Contact

For questions, please raise an [Issue] or email the corresponding author listed in the related publication.
