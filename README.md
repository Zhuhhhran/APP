[![DOI](https://zenodo.org/badge/1432793112.svg)](https://doi.org/10.5281/zenodo.23654828)
🚀 **Live Demo:** [Click here to use the AP-AKI Prediction Web App](https://a6tfdfcz5hbqvekfghk5rp.streamlit.app/)
# AP-AKI Prediction Model: An Explainable Machine Learning Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-orange)

This repository contains the source code for a web-based clinical prediction tool designed to estimate the risk of **Acute Kidney Injury (AKI) in patients with Acute Pancreatitis (AP)**. 

Built with **Streamlit** and **XGBoost**, this application allows clinicians to input raw laboratory values, automatically calculate complex metabolic and inflammatory indicators, and predict the likelihood of AP-AKI. Furthermore, it integrates **SHAP (SHapley Additive exPlanations)** to provide personalized, interpretable insights into each prediction.

## 🩺 Clinical Background

Acute Kidney Injury is a severe and common complication of Acute Pancreatitis, significantly increasing patient morbidity and mortality. Early identification of high-risk patients is crucial for timely clinical intervention. 

This model leverages key clinical features, including the **BISAP score** (Bedside Index for Severity in Acute Pancreatitis), **pleural effusion**, and novel metabolic/inflammatory markers like **TYG** (Triglyceride-glucose index), **AIP** (Atherogenic index of plasma), and **SII** (Systemic immune-inflammation index), to provide highly accurate risk stratification.

## 🚀 Features

* **Clinical Auto-Calculator**: The sidebar includes built-in calculators for composite indicators derived from raw laboratory values:
  * **TYG**: `Ln [TG (mmol/L) × GLU (mmol/L) / 2]`
  * **AIP**: `Log10 [TG (mmol/L) / HDL-C (mmol/L)]`
  * **SII**: `PLT (10^9/L) × N (10^9/L) / L (10^9/L)`
* **High-Performance Machine Learning**: Utilizes an optimized XGBoost binary classification model specifically trained on AP patient cohorts.
* **Explainable AI (XAI)**: Real-time SHAP force plots dynamically visualize how each of the 11 clinical features pushes the AKI risk up (red) or down (blue) for individual patients.
* **Clinical Performance**: The model demonstrates robust predictive power (Internal Validation AUC: 0.934, Accuracy: 87.0%, Sensitivity: 76.1%, Specificity: 93.8%).

## 📊 Input Features

The model requires 11 easily accessible clinical and laboratory variables:

1. **TYG** (Triglyceride-glucose index) - *Auto-calculated in sidebar*
2. **AIP** (Atherogenic index of plasma) - *Auto-calculated in sidebar*
3. **BISAP Score** (0-5)
4. **Pleural Effusion** (0 = No, 1 = Yes)
5. **BUN** (Blood Urea Nitrogen, mmol/L)
6. **SII** (Systemic immune-inflammation index) - *Auto-calculated in sidebar*
7. **PLT** (Platelet count, 10^9/L)
8. **D.D** (D-Dimer, ug/ml)
9. **Cys.C** (Cystatin C, mg/L)
10. **PT** (Prothrombin time, s)
11. **Hyperuricemia** (0 = No, 1 = Yes)

## 🛠️ Technology Stack

* **Web Framework**: Streamlit
* **Machine Learning**: XGBoost, Scikit-learn
* **Model Interpretability**: SHAP
* **Data Processing & Visualization**: Pandas, NumPy, Matplotlib

## 📂 Repository Structure

```text
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── artifacts/
    └── xgb_model.pkl       # Pre-trained XGBoost model (pickled)
