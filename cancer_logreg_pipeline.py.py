# cancer_logreg_pipeline.py
# Final, clean pipeline for Breast Cancer classification using Logistic Regression.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import joblib
import numpy as np

# ----------------------------
# 1) Load data
# ----------------------------
DATA_URL = "https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv"
if os.path.exists("Cancer.csv"):
    DATA_URL = "Cancer.csv"  # offline fallback

cancer = pd.read_csv(DATA_URL)

# ----------------------------
# 2) Quick checks
# ----------------------------
print("Shape:", cancer.shape)
print(cancer.head(3))
print(cancer.info())

# ----------------------------
# 3) Features/Target
# ----------------------------
# Target: 'diagnosis' ('M' malignant, 'B' benign)
y = cancer["diagnosis"]
# Drop id, diagnosis, and any unnamed column if present
X = cancer.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore")

# ----------------------------
# 4) Train/Test split (stratified)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=2529, stratify=y
)
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ----------------------------
# 5) Pipeline: scale + logistic regression
# ----------------------------
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")),
    ]
)

# ----------------------------
# 6) Fit
# ----------------------------
pipe.fit(X_train, y_train)

# ----------------------------
# 7) Predict + metrics
# ----------------------------
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]  # P(positive='M')

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# For ROC-AUC we need binary (M=1, B=0)
y_bin_test = (y_test == "M").astype(int)
auc = roc_auc_score(y_bin_test, y_prob)
print("\nROC-AUC:", f"{auc:.4f}")

# ----------------------------
# 8) Visuals: ROC + Confusion Matrix
# ----------------------------
RocCurveDisplay.from_predictions(y_bin_test, y_prob)
plt.title("Logistic Regression ROC Curve (Breast Cancer)")
plt.show()

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["B","M"])
plt.title("Confusion Matrix (labels: B=Benign, M=Malignant)")
plt.show()

# ----------------------------
# 9) (Optional) Cross-validated AUC
# ----------------------------
# Cross-validated AUC on the whole dataset for a more reliable estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2529)
# We need a binary scorer for AUC; scikit will infer positive class, but we ensure mapping explicitly:
y_bin_all = (y == "M").astype(int)
# Use a function to evaluate AUC with our pipeline:
def auc_cv_score(pipe, X, y_bin, cv):
    scores = []
    for train_idx, test_idx in cv.split(X, y_bin):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(roc_auc_score(y_bin.iloc[test_idx], prob))
    return np.array(scores)

cv_auc_scores = auc_cv_score(pipe, X, y_bin_all, cv)
print(f"\n5-fold CV ROC-AUC: mean={cv_auc_scores.mean():.4f}, std={cv_auc_scores.std():.4f}")

# ----------------------------
# 10) Save the model (pipeline)
# ----------------------------
joblib.dump(pipe, "cancer_logreg_pipeline.joblib")
print("\nSaved trained pipeline to: cancer_logreg_pipeline.joblib")

