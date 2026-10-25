import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# ---------- 页面配置 ----------
st.set_page_config(page_title="AP-AKI Prediction Model", layout="wide")

# ---------- 1. 加载 XGBoost 模型 ----------
@st.cache_resource
def load_model():
    # 去当前目录下的 artifacts 文件夹找 xgb_model.pkl
    path = Path(__file__).absolute().parent  / "xgb_model.pkl"
    
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"模型加载失败！请检查 artifacts 文件夹及文件名。\n错误信息：{e}")
    st.stop()

# ---------- 2. 变量与元数据 ----------
variable_names = [
    "TYG", "AIP", "BISAP", "Pleural.effusion", 
    "BUN", "SII", "PLT", "D.D", "Cys.C", 
    "PT", "hyperuricemia"
]

# 严格匹配你上传文件的单位
english_labels = {
    "TYG": "TYG (Calculated)",
    "AIP": "AIP (Calculated)",
    "BISAP": "BISAP Score",
    "Pleural.effusion": "Pleural Effusion",
    "BUN": "BUN (mmol/L)",
    "SII": "SII (Calculated)",
    "PLT": "PLT (10^9/L)",
    "D.D": "D-Dimer (ug/ml)",
    "Cys.C": "Cys-C (mg/L)",
    "PT": "PT (s)",
    "hyperuricemia": "Hyperuricemia"
}

# 二分类变量映射
category_labels = {
    "Pleural.effusion": {"0": "No", "1": "Yes"},
    "hyperuricemia": {"0": "No", "1": "Yes"},
}
categorical_vars = set(category_labels.keys())

# 默认值（严格按照你的基线表 Overall 列的中位数/众数）
defaults = {
    "TYG": 0.17, 
    "AIP": 3.27, 
    "BISAP": 1, 
    "Pleural.effusion": "0",
    "BUN": 4.80, 
    "SII": 1841.41, 
    "PLT": 233.00, 
    "D.D": 0.79, 
    "Cys.C": 0.79, 
    "PT": 12.30, 
    "hyperuricemia": "0",
}

feature_names = [english_labels[n] for n in variable_names]

# 初始化 session_state
for name in variable_names:
    if name not in st.session_state:
        st.session_state[name] = defaults[name]


# ==========================================
# 🧮 侧边栏：临床指标自动计算器
# ==========================================
with st.sidebar:
    st.header("🧮 Auto Calculator")
    st.markdown("Input raw labs to auto-calculate TYG, AIP, and SII.")
    
    # --- 1. TYG & AIP ---
    st.subheader("Lipid & Glucose")
    tg = st.number_input("TG (mmol/L)", value=3.47, step=0.1)
    glu = st.number_input("Fasting Glucose (mmol/L)", value=7.51, step=0.1)
    hdl = st.number_input("HDL-C (mmol/L)", value=1.26, step=0.1)
    
    # 计算 TYG: Ln [TG(mg/dL) * GLU(mg/dL) / 2]
    if tg > 0 and glu > 0:
        tyg_calc = np.log((tg * 88.57) * (glu * 18) / 2)
        st.session_state["TYG"] = tyg_calc
        st.caption(f"👉 Calculated TYG: **{tyg_calc:.2f}**")
        
    # 计算 AIP: Log10(TG / HDL-C)
    if tg > 0 and hdl > 0:
        aip_calc = np.log10(tg / hdl)
        st.session_state["AIP"] = aip_calc
        st.caption(f"👉 Calculated AIP: **{aip_calc:.2f}**")

    st.divider()

    # --- 2. SII & PLT ---
    st.subheader("Blood Routine")
    plt_val = st.number_input("PLT (10^9/L)", value=233.0, step=10.0)
    neu = st.number_input("Neutrophil (10^9/L)", value=8.90, step=0.1)
    lym = st.number_input("Lymphocyte (10^9/L)", value=1.13, step=0.1)
    
    if lym > 0:
        sii_calc = plt_val * neu / lym
        st.session_state["SII"] = sii_calc
        st.session_state["PLT"] = plt_val  # 同步更新主界面的 PLT
        st.caption(f"👉 Calculated SII: **{sii_calc:.2f}**")
    
    st.divider()
    
    # 【修改点】更新了这里的公式和单位说明，完全对齐你的 Word 文档
    with st.expander("📖 View formulas & units"):
        st.markdown("""
        **TYG (Triglyceride-glucose index)**
        *   Formula: `Ln [Fasting TG (mg/dL) × Fasting Glucose (mg/dL) / 2]`
        *   *Note: Inputs use **mmol/L**. The calculator converts TG (× 88.57) and GLU (× 18) to mg/dL automatically.*
        
        **AIP (Atherogenic index of plasma)**
        *   Formula: `Log10 [TG (mmol/L) / HDL-C (mmol/L)]`
        
        **SII (Systemic immune-inflammation index)**
        *   Formula: `PLT (10^9/L) × N (10^9/L) / L (10^9/L)`
        """)


# ==========================================
# 主界面：模型输入
# ==========================================
st.title("AKI Prediction Model")
st.markdown("The indicators (TYG, AIP, SII, PLT) are automatically updated by the **Calculator** on the left.")

col1, col2 = st.columns(2)

with col1:
    for name in variable_names[:6]:
        label = english_labels[name]
        if name in categorical_vars:
            st.selectbox(label, options=list(category_labels[name].keys()), format_func=lambda v, m=category_labels[name]: m[v], key=name)
        elif name == "BISAP":
            st.slider(label, min_value=0, max_value=5, step=1, key=name)
        else:
            # 自动计算的指标设为不可手动修改，避免数据冲突
            st.number_input(label, step=0.01, format="%.2f", key=name, disabled=(name in ["TYG", "AIP", "SII"]))

with col2:
    for name in variable_names[6:]:
        label = english_labels[name]
        if name in categorical_vars:
            st.selectbox(label, options=list(category_labels[name].keys()), format_func=lambda v, m=category_labels[name]: m[v], key=name)
        else:
            st.number_input(label, step=0.01, format="%.2f", key=name, disabled=(name == "PLT"))

# ---------- 4. 预测与 SHAP 解释 ----------
st.markdown("---")
if st.button("Predict", type="primary", use_container_width=True):
    # 从 session_state 提取 11 个变量
    X = np.array([[float(st.session_state[n]) for n in variable_names]], dtype=float)

    proba = model.predict_proba(X)[0]          
    classes = list(model.classes_)
    prob_positive = proba[classes.index(1)] if 1 in classes else proba[1]
    
    st.markdown("### Prediction Result")
    if prob_positive >= 0.5:
        st.error(f"**High Risk of AKI** - Probability: {prob_positive:.2%}")
    else:
        st.success(f"**Low Risk of AKI** - Probability: {prob_positive:.2%}")

    # SHAP 力图
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)

    sv = np.asarray(explanation.values)
    bv = np.asarray(explanation.base_values)

    shap_values = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    base_value = float(bv[0, 1]) if bv.ndim == 2 else float(bv[0])

    st.subheader("Individual SHAP Analysis")
    st.markdown("The plot below shows how each feature pushes the AKI risk up (red) or down (blue):")
    
    if not np.allclose(shap_values, 0):
        fig = shap.force_plot(base_value, shap_values, X[0], feature_names=feature_names, matplotlib=True, show=False, figsize=(16, 4))
        st.pyplot(fig)
        plt.close(fig)

    # 页面底部展示模型性能
    st.caption("**Model Performance (Internal Validation):** AUC 0.934 | Accuracy 87.0% | Sensitivity 76.1% | Specificity 93.8%")