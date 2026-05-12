# 🌸 Iris Flower Classifier — Devoir ML dans le Cloud

Projet de Machine Learning déployé sur le cloud avec **Streamlit** et **Render**.

## 📁 Structure du projet

```
ml_project/
├── app.py                  # Application Streamlit (interface utilisateur)
├── train_model.py          # Script d'entraînement du modèle
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Configuration Docker (optionnel)
├── deployment_link.txt     # Lien de déploiement sur Render
├── README.md               # Ce fichier
└── model/                  # Modèle entraîné (généré automatiquement)
    ├── iris_model.pkl
    ├── iris_scaler.pkl
    └── model_meta.json
```

## 🤖 Modèle

| Paramètre       | Valeur                     |
|-----------------|----------------------------|
| Dataset         | Iris (Fisher, 1936)        |
| Algorithme      | Random Forest Classifier   |
| Estimateurs     | 100 arbres                 |
| Max depth       | 5                          |
| Accuracy        | 93.33%                     |
| Classes         | setosa, versicolor, virginica |

## 🚀 Lancement local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle
python train_model.py

# 3. Lancer l'application Streamlit
streamlit run app.py
```

## 🐳 Docker

```bash
# Build
docker build -t iris-classifier .

# Run
docker run -p 8501:8501 iris-classifier
```

## ☁️ Déploiement sur Render

1. Pusher le projet sur GitHub
2. Sur [render.com](https://render.com), créer un **Web Service**
3. Connecter le dépôt GitHub
4. Configurer :
   - **Build Command** : `pip install -r requirements.txt && python train_model.py`
   - **Start Command** : `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Déployer → copier le lien dans `deployment_link.txt`

## 📊 Dataset

Le dataset **Iris** est un classique du Machine Learning :
- **150 échantillons** répartis en 3 classes de 50 chacune
- **4 features** : longueur/largeur du sépale et du pétale (en cm)
- **3 classes** : Iris setosa, Iris versicolor, Iris virginica

---
*Devoir Machine Learning dans le Cloud — Random Forest + Streamlit + Render*
