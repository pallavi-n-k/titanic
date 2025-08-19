#!/usr/bin/env python3
"""
Titanic Data Analysis — End-to-End Script (Standalone)
- Loads data (Seaborn Titanic or titanic.csv fallback)
- Cleans & engineers features
- Generates figures (displayed, not saved)
- Trains Logistic Regression and Random Forest
- Displays metrics, ROC curves, confusion matrices, and metrics summary

Run: python titanic_analysis.py
Requires: pandas numpy matplotlib scikit-learn (and seaborn for built-in dataset, optional)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay
)

# -----------------------------
# 1) Load data
# -----------------------------
df = None
source = None
try:
    import seaborn as sns  # only for dataset loading
    df = sns.load_dataset("titanic")
    source = "seaborn.load_dataset('titanic')"
except Exception:
    csv_candidates = [
        r"C:\\Users\\Lenovo\\Desktop\\titanic.csv.xlsx",  # Your file path
        "titanic.csv",
        os.path.join("data", "titanic.csv"),
        os.path.join("dataset", "titanic.csv"),
    ]
    for p in csv_candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            source = f"local_csv:{p}"
            break

if df is None:
    raise FileNotFoundError(
        "Could not load dataset. Install seaborn to use its built-in Titanic dataset "
        "or place 'titanic.csv' in the project root or data/ or dataset/ folder."
    )

# Normalize column names if using a Kaggle-style CSV
rename_map = {
    'Pclass': 'pclass', 'Sex': 'sex', 'Age': 'age', 'SibSp': 'sibsp',
    'Parch': 'parch', 'Fare': 'fare', 'Embarked': 'embarked',
    'Survived': 'survived', 'Cabin': 'deck'
}
df = df.rename(columns=rename_map)

# -----------------------------
# 2) Basic cleaning & feature engineering
# -----------------------------
if 'deck' in df.columns:
    df = df.drop(columns=['deck'])  # very sparse

# Family size & IsAlone
if 'sibsp' in df.columns and 'parch' in df.columns:
    df['family_size'] = df['sibsp'].fillna(0) + df['parch'].fillna(0) + 1
else:
    df['family_size'] = 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

target_col = 'survived'
feature_cols = ['pclass', 'sex', 'age', 'fare', 'embarked', 'family_size', 'is_alone']
available_features = [c for c in feature_cols if c in df.columns]

df = df.dropna(subset=[target_col])
X = df[available_features]
y = df[target_col].astype(int)

# -----------------------------
# 3) Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4) Preprocessing pipelines
# -----------------------------
num_features = [c for c in ['age', 'fare', 'family_size'] if c in X.columns]
cat_features = [c for c in ['pclass', 'sex', 'embarked', 'is_alone'] if c in X.columns]

numeric_tf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_tf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_tf, num_features),
        ('cat', categorical_tf, cat_features)
    ]
)

# -----------------------------
# 5) Models
# -----------------------------
logreg = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', LogisticRegression(max_iter=1000))
])

rf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ))
])

# -----------------------------
# 6) Fit & Evaluate
# -----------------------------
def evaluate_and_plot(name, clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = float('nan')

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )



# -----------------------------
# 7) Simple EDA figures (matplotlib only)
# -----------------------------
def safe_plot_bar_counts(series, title):
    counts = series.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(series.name if series.name else "")
    ax.set_ylabel("Count")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def safe_hist(series, title, bins=30):
    data = series.dropna().values
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name if series.name else "")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# Survival distribution
if 'survived' in df.columns:
    safe_plot_bar_counts(df['survived'], "Survival Distribution")

# Survival by Sex (grouped bars)
if {'sex', 'survived'}.issubset(df.columns):
    pivot = pd.crosstab(df['sex'], df['survived'])
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = np.arange(len(pivot.index))
    ax.bar(x - width/2, pivot[0].values, width, label='Not Survived')
    ax.bar(x + width/2, pivot[1].values, width, label='Survived')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str))
    ax.set_title("Survival by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# Survival by Class (grouped bars)
if {'pclass', 'survived'}.issubset(df.columns):
    pivot = pd.crosstab(df['pclass'], df['survived'])
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = np.arange(len(pivot.index))
    ax.bar(x - width/2, pivot[0].values, width, label='Not Survived')
    ax.bar(x + width/2, pivot[1].values, width, label='Survived')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str))
    ax.set_title("Survival by Passenger Class")
    ax.set_xlabel("Pclass")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# Age distribution
if 'age' in df.columns:
    safe_hist(df['age'], "Age Distribution", bins=30)

print(f"\nAnalysis completed. Data source: {source}")
