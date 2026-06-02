"""
API Performans Yönetimi - Streamlit Dashboard
Çalıştırmak için: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="API Performans Dashboard", page_icon="📊", layout="wide")

FEATURES = [
    'payload_size', 'active_requests', 'rps_10s', 'requests_last_60s',
    'response_time_lag_1', 'response_time_lag_2', 'response_time_lag_3',
    'response_time_roll_mean_5', 'response_time_roll_mean_10',
    'response_time_roll_std_5', 'endpoint_encoded'
]
TARGET      = 'response_time_ms'
WINDOW_SIZE = 10
RISK_COLORS = {"Normal": "#2ecc71", "Yoğun": "#f39c12", "Kritik": "#e74c3c"}

def classify_risk(ms):
    if ms < 500:    return "Normal"
    elif ms < 1000: return "Yoğun"
    return "Kritik"

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data/raw/request_level_data.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_resource
def load_model():
    path = os.path.join(BASE, "models/random_forest_enhanced.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_encoder():
    path = os.path.join(BASE, "models/label_encoder.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_results():
    rp = os.path.join(BASE, "data/risk_results.csv")
    sp = os.path.join(BASE, "data/shap_summary.csv")
    return (pd.read_csv(rp) if os.path.exists(rp) else None,
            pd.read_csv(sp) if os.path.exists(sp) else None)

df       = load_data()
rf_model = load_model()
le       = load_encoder()
risk_df, shap_df = load_results()

st.title("📊 API Performans Yönetimi")
st.markdown("**Öngörücü Analitik ve Açıklanabilir Yapay Zeka Tabanlı Sistem**")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Genel Bakış", "🎯 Tahmin & Risk", "🔍 SHAP Analizi", "⚙️ What-If Analizi"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Temel Metrikler")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam İstek",         f"{len(df):,}")
    c2.metric("Ort. Response Time",   f"{df[TARGET].mean():.1f} ms")
    c3.metric("Medyan Response Time", f"{df[TARGET].median():.1f} ms")
    c4.metric("Maks. Response Time",  f"{df[TARGET].max():.0f} ms")
    st.markdown("---")

    st.subheader("Response Time - Zaman Serisi")
    n = st.slider("Kayıt sayısı", 500, 5000, 2000, step=500)
    fig, ax = plt.subplots(figsize=(12, 4))
    s = df[TARGET].iloc[:n]
    ax.plot(s.values, linewidth=0.8, alpha=0.7, color="#3498db", label="Response Time")
    ax.plot(s.rolling(50).mean().values, linewidth=2, color="#e74c3c", label="Rolling Mean (50)")
    ax.axhline(500,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.6, label="Yoğun eşiği")
    ax.axhline(1000, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6, label="Kritik eşiği")
    ax.set_xlabel("Sample"); ax.set_ylabel("ms"); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

    st.subheader("Senaryolara Göre Dağılım")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    scenarios = sorted(df["scenario"].unique())
    ax2.boxplot([df[df["scenario"]==s][TARGET].values for s in scenarios], labels=scenarios, patch_artist=True)
    ax2.set_ylabel("ms"); ax2.grid(alpha=0.3)
    st.pyplot(fig2); plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Performans Karşılaştırması")
    cmp = pd.DataFrame([
        {"Model": "LR (baseline)",   "MAE": 147.21, "RMSE": 196.44, "R²": 0.024},
        {"Model": "RF (baseline)",   "MAE": 146.37, "RMSE": 196.56, "R²": 0.023},
        {"Model": "LSTM (baseline)", "MAE": 121.23, "RMSE": 177.79, "R²": 0.201},
        {"Model": "LR (enhanced)",   "MAE": 123.71, "RMSE": 178.91, "R²": 0.191},
        {"Model": "RF (enhanced)",   "MAE":  88.17, "RMSE": 152.82, "R²": 0.410},
        {"Model": "LSTM (enhanced)", "MAE":  88.75, "RMSE": 154.16, "R²": 0.399},
    ]).set_index("Model")
    st.dataframe(cmp.style
                 .highlight_min(axis=0, subset=["MAE","RMSE"], color="#d5f5e3")
                 .highlight_max(axis=0, subset=["R²"],         color="#d5f5e3"))
    st.markdown("---")

    if risk_df is not None:
        c1, c2 = st.columns(2)
        for col, rcol, title in [(c1,"risk_true","Gerçek"), (c2,"risk_pred","Tahmin Edilen")]:
            with col:
                st.markdown(f"**{title} Risk Dağılımı**")
                counts = risk_df[rcol].value_counts().reindex(["Normal","Yoğun","Kritik"], fill_value=0)
                fig, ax = plt.subplots(figsize=(5, 4))
                bars = ax.bar(counts.index, counts.values,
                              color=[RISK_COLORS[r] for r in counts.index], alpha=0.85)
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                            str(val), ha='center', fontsize=10)
                ax.grid(alpha=0.3, axis='y')
                st.pyplot(fig); plt.close()

        st.subheader("Gerçek vs Tahmin (İlk 300 Sample)")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(risk_df["y_true"].values[:300],    label="Gerçek",      linewidth=1.5, alpha=0.85)
        ax.plot(risk_df["y_pred_rf"].values[:300], label="RF Enhanced", linewidth=1.5, alpha=0.85)
        ax.axhline(500,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.7, label="Yoğun")
        ax.axhline(1000, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="Kritik")
        ax.set_xlabel("Sample"); ax.set_ylabel("ms"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()
    else:
        st.warning("Risk sonuçları bulunamadı. Önce 04-risk-and-shap-v2.ipynb çalıştırın.")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("SHAP - Açıklanabilir Yapay Zeka")
    if shap_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            sorted_s = shap_df.sort_values("mean_abs_shap", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.barh(sorted_s["feature"], sorted_s["mean_abs_shap"], color="#3498db", alpha=0.85)
            for bar, val in zip(bars, sorted_s["mean_abs_shap"]):
                ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va='center', fontsize=9)
            ax.set_xlabel("Ortalama |SHAP|"); ax.set_title("SHAP Feature Importance"); ax.grid(alpha=0.3, axis='x')
            st.pyplot(fig); plt.close()
        with c2:
            st.info("**Pozitif SHAP** → response time artıyor\n\n**Negatif SHAP** → response time azalıyor\n\n**Büyük mutlak değer** → güçlü etki")
            for _, row in shap_df.iterrows():
                st.markdown(f"- **{row['feature']}**: {row['mean_abs_shap']:.2f}")

        for img, cap in [("outputs/shap_beeswarm.png","SHAP Beeswarm"),
                          ("outputs/shap_waterfall.png","SHAP Waterfall")]:
            p = os.path.join(BASE, img)
            if os.path.exists(p):
                st.subheader(cap); st.image(p, use_column_width=True)
    else:
        st.warning("SHAP sonuçları bulunamadı. Önce 04-risk-and-shap-v2.ipynb çalıştırın.")

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("⚙️ What-If Analizi")
    if rf_model is not None and le is not None:
        c1, c2 = st.columns(2)
        with c1:
            payload_size      = st.slider("Payload Size (bytes)",  10, 2000,  300, step=10)
            active_requests   = st.slider("Active Requests",        1,   80,   10, step=1)
            rps_10s           = st.slider("RPS (son 10s)",          1,  500,   50, step=5)
            requests_last_60s = st.slider("Requests (son 60s)",     1,  700,  150, step=10)
            endpoint_name     = st.selectbox("Endpoint", list(le.classes_))
            endpoint_encoded  = int(le.transform([endpoint_name])[0])
            st.markdown("**Geçmiş Gecikme (ms)**")
            lag1 = st.number_input("t-1", min_value=0, value=300)
            lag2 = st.number_input("t-2", min_value=0, value=280)
            lag3 = st.number_input("t-3", min_value=0, value=260)

        with c2:
            roll_mean_5  = np.mean([lag1, lag2, lag3, 270, 265])
            roll_mean_10 = np.mean([lag1, lag2, lag3, 270, 265, 260, 255, 250, 245, 240])
            roll_std_5   = np.std([lag1, lag2, lag3, 270, 265])

            row = np.array([[payload_size, active_requests, rps_10s, requests_last_60s,
                             lag1, lag2, lag3, roll_mean_5, roll_mean_10, roll_std_5,
                             endpoint_encoded]])
            window = np.tile(row, (1, WINDOW_SIZE))

            pred_ms    = rf_model.predict(window)[0]
            risk_level = classify_risk(pred_ms)
            rc         = RISK_COLORS[risk_level]

            st.markdown(f"""
            <div style="background:{rc}22;border-left:5px solid {rc};padding:20px;border-radius:5px">
                <h2 style="color:{rc};margin:0">⏱ {pred_ms:.0f} ms</h2>
                <h3 style="color:{rc};margin:5px 0 0 0">🚦 {risk_level}</h3>
            </div>""", unsafe_allow_html=True)

            if risk_level == "Normal":   st.success("✅ Sistem stabil.")
            elif risk_level == "Yoğun": st.warning("⚠️ Yük artmış, izleme önerilir.")
            else:                        st.error("🔴 Kritik! Müdahale gerekebilir.")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(["Tahmin"], [pred_ms], color=rc, alpha=0.85)
            ax.axvline(500,  color="#f39c12", linestyle="--", linewidth=1.5, label="Yoğun")
            ax.axvline(1000, color="#e74c3c", linestyle="--", linewidth=1.5, label="Kritik")
            ax.set_xlim(0, max(1200, pred_ms*1.2)); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')
            st.pyplot(fig); plt.close()
    else:
        st.warning("Model bulunamadı. Önce 02b-enhanced-model.ipynb çalıştırın.")

st.markdown("---")
st.caption("API Performans Yönetimi | Bahar EROL | Samsun Üniversitesi 2026")
