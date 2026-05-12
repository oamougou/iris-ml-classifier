# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Iris ML Classifier (Streamlit)
# Base image : Python 3.11 slim
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Devoir ML Cloud"
LABEL description="Iris Flower Classifier — Random Forest + Streamlit"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Répertoire de travail
WORKDIR /app

# Copie des dépendances en premier (optimisation du cache Docker)
COPY requirements.txt .

# Installation des dépendances système (légères)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie de tout le projet
COPY . .

# Entraînement du modèle au build (génère model/)
RUN python train_model.py

# Exposition du port Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande de démarrage
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
