# Imports 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, 
                             classification_report, 
                             roc_curve, 
                             auc)
import matplotlib.pyplot as plt

# Data Loading & Prepossesing
def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Reads the CSV, cleans types, handles missing values,
    and encodes categorical variables.
    """
    df = pd.read_csv(path)
    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df[col] = df[col].astype("category").cat.codes
    return df

# Standardize, split, and return
def split_and_scale(df: pd.DataFrame):
    """
    Splits features/target, train/test, and scales features.
    Returns scaled arrays + raw splits as needed for plotting.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, X_train, X_test

# Generic trainer / evaluator
def evaluate_model(model, X_train, X_test, y_train, y_test, name:str):
    """
    Fits the model, prints metrics, and returns AUC curve values for plotting.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    # ROC curve
    fpr = tpr = None
    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"ROC-AUC: {roc_auc:.4f}")

    return fpr, tpr, roc_auc

# Main Pipeline
if __name__ == "__main__":
    df = load_and_preprocess("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X_train_s, X_test_s, y_train, y_test, X_train, X_test = split_and_scale(df)

    lr = LogisticRegression(max_iter=1000)
    lr_fpr, lr_tpr, lr_auc = evaluate_model(lr, X_train_s, X_test_s, y_train, y_test, "Logistic Regression")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_fpr, rf_tpr, rf_auc = evaluate_model(rf, X_train_s, X_test_s, y_train, y_test, "Random Forest")

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_fpr, xgb_tpr, xgb_auc = evaluate_model(xgb, X_train_s, X_test_s, y_train, y_test, "XGBoost")

    plt.figure()
    if lr_fpr is not None:  plt.plot(lr_fpr,  lr_tpr,  label=f"LR (AUC={lr_auc:.2f})")
    if rf_fpr is not None:  plt.plot(rf_fpr,  rf_tpr,  label=f"RF (AUC={rf_auc:.2f})")
    if xgb_fpr is not None: plt.plot(xgb_fpr, xgb_tpr, label=f"XGB (AUC={xgb_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # diagonal line = random guess
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()