"""
Application Streamlit - Classificateur de Fleurs Iris
Modèle : Random Forest | Dataset : Iris (UCI/Sklearn)
Auteur  : Devoir ML dans le Cloud
"""

import streamlit as st
import numpy as np
import joblib
import json
import os
import time

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier | ML Cloud",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 {
    color: #fff;
    font-size: 2.6rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.65);
    margin: 0.5rem 0 0;
    font-size: 1rem;
}

/* Metric cards */
.metric-card {
    background: #fff;
    border: 1px solid #e8e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #0f2027;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* Prediction result */
.result-setosa {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-left: 5px solid #2e7d32;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-versicolor {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 5px solid #1565c0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-virginica {
    background: linear-gradient(135deg, #fce4ec, #f8bbd9);
    border-left: 5px solid #880e4f;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.result-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    margin: 0;
}
.result-desc {
    font-size: 0.9rem;
    color: #555;
    margin-top: 0.4rem;
}

/* Probability bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}
.prob-label {
    font-size: 0.82rem;
    width: 100px;
    color: #444;
}
.prob-bar-bg {
    background: #e8e8f0;
    border-radius: 6px;
    height: 10px;
    flex: 1;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-size: 0.82rem;
    width: 45px;
    text-align: right;
    font-weight: 600;
    color: #222;
}

/* Info box */
.info-box {
    background: #f8f9ff;
    border: 1px solid #e0e4ff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.88rem;
    color: #444;
    line-height: 1.7;
}

/* Sidebar */
.sidebar-section {
    background: #f0f2f6;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.87rem;
    color: #000000;
}
.sidebar-section strong {
    color: #000000;
}
.sidebar-section br + * {
    color: #000000;
}
</style>
""", unsafe_allow_html=True)


# ─── Chargement du modèle ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model/iris_model.pkl")
    scaler = joblib.load("model/iris_scaler.pkl")
    with open("model/model_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta

try:
    model, scaler, meta = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    st.stop()


# ─── Description des classes ──────────────────────────────────────────────────
CLASS_INFO = {
    "setosa": {
        "emoji": "🌿",
        "color": "#2e7d32",
        "css": "result-setosa",
        "desc": "Iris setosa — Petits pétales très distincts. Facile à identifier grâce à ses dimensions compactes.",
        "origin": "Arctique / subarctique"
    },
    "versicolor": {
        "emoji": "💙",
        "color": "#1565c0",
        "css": "result-versicolor",
        "desc": "Iris versicolor — Fleur de taille intermédiaire aux teintes violacées. Très répandue en Amérique du Nord.",
        "origin": "Amérique du Nord"
    },
    "virginica": {
        "emoji": "🌸",
        "color": "#880e4f",
        "css": "result-virginica",
        "desc": "Iris virginica — La plus grande des trois espèces, aux pétales longs et larges.",
        "origin": "Est des États-Unis"
    }
}

BAR_COLORS = {
    "setosa": "#2e7d32",
    "versicolor": "#1565c0",
    "virginica": "#880e4f"
}

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌸 Iris Flower Classifier</h1>
    <p>Modèle de Machine Learning déployé sur le Cloud · Random Forest · Dataset UCI Iris</p>
</div>
""", unsafe_allow_html=True)

# ─── Métriques du modèle ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{meta['accuracy']*100:.1f}%</div>
        <div class="label">Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{meta['n_estimators']}</div>
        <div class="label">Arbres (RF)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{meta['train_size']}</div>
        <div class="label">Train samples</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">3</div>
        <div class="label">Classes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Layout principal ─────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 🔬 Saisir les mesures de la fleur")
    st.markdown('<div class="info-box">Ajustez les 4 mesures morphologiques de la fleur (en centimètres) pour obtenir une prédiction.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    sepal_length = st.slider("📏 Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("📐 Largeur du sépale (cm)",  2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("📏 Longueur du pétale (cm)", 1.0, 7.0, 4.4, 0.1)
    petal_width  = st.slider("📐 Largeur du pétale (cm)",  0.1, 2.5, 1.3, 0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bouton de prédiction
    predict_btn = st.button("🚀  Classifier la fleur", type="primary", use_container_width=True)

    # Exemples rapides
    st.markdown("#### Exemples rapides")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("🌿 Setosa", use_container_width=True):
            st.session_state["ex"] = [5.1, 3.5, 1.4, 0.2]
    with ex2:
        if st.button("💙 Versicolor", use_container_width=True):
            st.session_state["ex"] = [6.0, 2.9, 4.5, 1.5]
    with ex3:
        if st.button("🌸 Virginica", use_container_width=True):
            st.session_state["ex"] = [6.7, 3.0, 5.2, 2.3]

    if "ex" in st.session_state:
        ex = st.session_state["ex"]
        st.info(f"💡 Exemple chargé : {ex} — Cliquez sur **Classifier** pour voir le résultat.")

with col_result:
    st.markdown("### 📊 Résultat de la classification")

    # Récupère les valeurs d'exemple si disponibles
    if "ex" in st.session_state and not predict_btn:
        vals = st.session_state["ex"]
        sepal_length, sepal_width, petal_length, petal_width = vals
        run_predict = True
    elif predict_btn:
        run_predict = True
    else:
        run_predict = False

    if run_predict:
        X_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        X_scaled = scaler.transform(X_input)

        with st.spinner("Analyse en cours..."):
            time.sleep(0.4)
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]

        class_names = meta["classes"]
        pred_name = class_names[prediction]
        info = CLASS_INFO[pred_name]

        # Résultat principal
        st.markdown(f"""
        <div class="{info['css']}">
            <p class="result-name">{info['emoji']}  Iris <em>{pred_name}</em></p>
            <p class="result-desc">{info['desc']}</p>
            <small>🌍 Origine : {info['origin']}</small>
        </div>
        """, unsafe_allow_html=True)

        # Probabilités
        st.markdown("<br>**Probabilités par classe**", unsafe_allow_html=True)
        for i, cls in enumerate(class_names):
            pct = probabilities[i] * 100
            color = BAR_COLORS[cls]
            st.markdown(f"""
            <div class="prob-row">
                <span class="prob-label">Iris {cls}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Valeurs saisies
        st.markdown("<br>**Mesures utilisées**")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.metric("Longueur sépale", f"{sepal_length} cm")
            st.metric("Longueur pétale", f"{petal_length} cm")
        with feat_col2:
            st.metric("Largeur sépale", f"{sepal_width} cm")
            st.metric("Largeur pétale", f"{petal_width} cm")

    else:
        st.markdown("""
        <div style="border: 2px dashed #dde; border-radius: 12px; padding: 3rem; text-align: center; color: #aaa; margin-top: 1rem;">
            <div style="font-size: 3rem;">🔬</div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">Ajustez les mesures et cliquez sur<br><strong>Classifier la fleur</strong></div>
        </div>
        """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ À propos")
    st.markdown(f"""
    <div class="sidebar-section">
    <strong>📁 Dataset</strong><br>
    Iris Dataset (Fisher, 1936)<br>
    150 échantillons · 4 features · 3 classes<br><br>
    <strong>🤖 Modèle</strong><br>
    Random Forest Classifier<br>
    {meta['n_estimators']} estimateurs · max_depth={meta['max_depth']}<br><br>
    <strong>✅ Accuracy</strong><br>
    {meta['accuracy']*100:.2f}% sur le jeu de test
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📚 Features")
    st.markdown("""
    <div class="sidebar-section">
    1. <strong>Sepal Length</strong> — Longueur du sépale<br>
    2. <strong>Sepal Width</strong> — Largeur du sépale<br>
    3. <strong>Petal Length</strong> — Longueur du pétale<br>
    4. <strong>Petal Width</strong> — Largeur du pétale
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🌸 Classes")
    for name, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **Iris {name}** — {info['origin']}")

    st.markdown("---")
    st.caption("Déployé avec Streamlit + Render · ML Cloud Devoir")
