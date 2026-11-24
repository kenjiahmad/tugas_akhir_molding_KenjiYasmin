# Injection Molding Defect Prediction System

An end-to-end machine learning application designed to predict product defect risks during injection molding production cycles, served via a FastAPI REST API and an interactive Streamlit web interface.

- **Live Application:** https://tugasakhirmoldingkenjiyasmin-qyqifvknuvkdqkyvkzgmza.streamlit.app/
- **Demo Walkthrough:** https://drive.google.com/drive/folders/1rk7Q_rQ-CoNzu9vFC217uI51LrfgHR_X?usp=sharing *(Audio in Indonesian)*

## Problem Statement & Target Users
Built for **plant managers** and **quality control (QC) operators** on injection molding production lines. The model predicts defect probability per molding cycle before items move to the next production stage, enabling targeted manual inspection[cite: 5]. By prioritizing high-risk items, the system minimizes operational losses from undetected defects (**False Negative cost = IDR 500,000**) relative to re-inspection costs (**False Positive cost = IDR 25,000**)[cite: 5].

## Data Cleaning & Preprocessing Pipeline
| Data Issue Identified | Affected Rows | Handling Strategy |
| :--- | :--- | :--- |
| Exact duplicate records | 350 | Removed via `drop_duplicates()`, retaining 25,000 clean rows[cite: 5] |
| Inconsistent categorical text (`shift` & `bahan`) | All text columns | Standardized using `str.strip()` and `lower()` / `upper()`[cite: 5] |
| Missing values in `waktu_pendinginan_s` (cooling time) | 257 | Imputed using median values via `SimpleImputer` inside the pipeline[cite: 5] |
| Physically impossible values (`tekanan_injeksi_bar <= 0`) | 125 | Converted to `NaN` and median-imputed inside the pipeline[cite: 5] |
| Physically impossible values (`waktu_siklus_s <= 0`) | 75 | Converted to `NaN` and median-imputed inside the pipeline[cite: 5] |

## Model Evaluation & Cost Optimization
Due to class imbalance, models were evaluated using **Precision-Recall AUC (PR-AUC)**[cite: 5]:

| Model | Validation PR-AUC | Test PR-AUC |
| :--- | :--- | :--- |
| *Baseline:* Dummy Random | 0.0748 | -[cite: 5] |
| *Baseline:* Logistic Regression | 0.4989 | -[cite: 5] |
| Random Forest | 0.5913 | -[cite: 5] |
| **XGBoost (Final Model)** | **0.6194** | **0.6504**[cite: 5] |

**Threshold Selection (`0.23`):** Selected via cost-based optimization assuming an FN loss of IDR 500,000 and an FP inspection cost of IDR 25,000, minimizing total operational cost while capturing **87.92%** of defective products on the test set[cite: 5].

## Tech Stack & Architecture
- **Machine Learning:** Python, Scikit-Learn, XGBoost, Pandas, NumPy[cite: 5, 6]
- **Backend & Deployment:** FastAPI (REST API), Streamlit Cloud
