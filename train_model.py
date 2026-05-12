"""
Script d'entraînement du modèle ML
Dataset: Iris (classification de fleurs)
Modèle: Random Forest Classifier
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ─── Chargement du dataset ─────────────────────────────────────────────────────
print("=" * 60)
print("  ENTRAÎNEMENT DU MODÈLE - CLASSIFICATION IRIS")
print("=" * 60)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(f"\n✅ Dataset chargé : {X.shape[0]} échantillons, {X.shape[1]} features")
print(f"   Classes : {list(iris.target_names)}")
print(f"\n   Aperçu des données :")
print(X.head(3).to_string())

# ─── Séparation train/test ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Split : {len(X_train)} train | {len(X_test)} test")

# ─── Normalisation ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Entraînement du modèle ────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("\n✅ Modèle entraîné (Random Forest - 100 arbres)")

# ─── Évaluation ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'─'*60}")
print(f"  RÉSULTATS")
print(f"{'─'*60}")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"\n  Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─── Sauvegarde ───────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,  "model/iris_model.pkl")
joblib.dump(scaler, "model/iris_scaler.pkl")

# Sauvegarde des métadonnées
import json
meta = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 100,
    "max_depth": 5,
    "accuracy": round(acc, 4),
    "features": iris.feature_names,
    "classes": list(iris.target_names),
    "train_size": len(X_train),
    "test_size": len(X_test)
}
with open("model/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Modèle sauvegardé dans model/iris_model.pkl")
print("✅ Scaler sauvegardé dans model/iris_scaler.pkl")
print("✅ Métadonnées sauvegardées dans model/model_meta.json")
print("\n🎉 Entraînement terminé avec succès !")
