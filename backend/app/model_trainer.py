# type: ignore — sklearn/imblearn have incomplete type stubs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import shap
import joblib
import json
import os


FEATURE_COLUMNS = [
    # External credit scores
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    # Time-based and regional features
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "FLAG_EMP_PHONE",
    "FLAG_DOCUMENT_3",
    # Financial amounts (domain knowledge)
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
]

TARGET_COLUMN = "TARGET"


def load_and_prepare_data(filepath: str):

    print("Loading dataset....")
    df = pd.read_csv(filepath)
    print(f"Row data shape: {df.shape}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    print(f"After feature selection shape: {df.shape}")

    missing = df.isnull().mean() * 100
    print(f"\nMissing value:\n{missing[missing > 0].to_string()}")

    df = df.fillna(df.median(numeric_only=True))

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"\nTarget distribution:\n{y.value_counts(normalize=True).to_string()}")
    return X, y


def split_and_resample(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set : {X_test.shape[0]} rows")
    print("\nBefore SMOTE training TARGET distribution: ")
    print(y_train.value_counts().to_string())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(np.array(X_train), np.array(y_train))

    print("\nAfter SMOTE — training target distribution:")
    print(pd.Series(y_resampled).value_counts().to_string())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(np.array(X_test))

    return X_train_scaled, X_test_scaled, y_resampled, y_test, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=11,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
        }

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    return models, results


def compute_shap_values(model, X_test):

    print("\nComputing SHAP values (this may take a minute)...")
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        background = shap.kmeans(X_test, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        feature_importance[col] = round(float(mean_shap[i]), 6)

    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )
    print("\nFeature Importance (SHAP):")
    for feat, val in feature_importance.items():
        print(f"  {feat}: {val}")

    return feature_importance


def save_artifacts(bast_model, scaler, results, feature_importance, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(bast_model, os.path.join(model_dir, "best_model.joblib"))
    print(f"\nModel saved to {model_dir}/best_model.joblib")

    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    print(f"\nScaler saved to {model_dir}/scaler.joblib")

    metadata = {
        "model_results": results,
        "feature_importance": feature_importance,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {model_dir}/metadata.json")


def select_best_model(models, results):

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = models[best_name]

    print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']})")

    return best_name, best_model


def main():

    filepath = os.path.join("data", "application_train.csv")
    X, y = load_and_prepare_data(filepath)

    X_train, X_test, y_train, y_test, scaler = split_and_resample(X, y)

    models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    best_name, best_model = select_best_model(models, results)

    feature_importance = compute_shap_values(best_model, X_test)

    save_artifacts(best_model, scaler, results, feature_importance)

    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 50)
    print(f"\nBest model: {best_name}")
    print(f"ROC-AUC:    {results[best_name]['roc_auc']}")
    print(f"Recall:     {results[best_name]['recall']}")
    print(f"F1 Score:   {results[best_name]['f1_score']}")
    print("\nSaved artifacts in /models directory:")
    print("  - best_model.joblib")
    print("  - scaler.joblib")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
