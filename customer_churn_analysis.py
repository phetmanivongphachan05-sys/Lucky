import argparse
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


TARGET_CANDIDATES = [
    "churn",
    "Churn",
    "churned",
    "is_churn",
    "is_churned",
    "customer_churn",
    "CustomerChurn",
    "target",
    "y"
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dữ liệu không tìm thấy: {path}")

    df = pd.read_csv(path)
    print(f"Loaded data: {path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df


def find_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Không tìm thấy cột churn trong dữ liệu. Vui lòng cung cấp dữ liệu có cột churn/churned/target."
    )


def print_basic_info(df: pd.DataFrame, target_col: str) -> None:
    print("\n=== Thông tin chung ===")
    print(df.head(5).to_string(index=False))
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values: \n", df.isna().sum().sort_values(ascending=False).head(20))
    print("\nTarget distribution:")
    print(df[target_col].value_counts(dropna=False))


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [col for col in df.columns if "date" in col.lower() or "day" in col.lower()]
    if not date_cols:
        return df

    today = pd.Timestamp(datetime.today().date())
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_age_days"] = (today - df[col]).dt.days
        except Exception:
            continue

    return df


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = extract_date_features(df.copy())

    drop_columns = []
    for col in df.columns:
        if col.lower() in {"customer_id", "id", "customerid", "userid", "order_id", "orderid"}:
            drop_columns.append(col)
        if col == target_col:
            continue
        if col.lower().endswith("date") or col.lower().endswith("datetime"):
            drop_columns.append(col)

    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col], errors="ignore")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    return X, y, numeric_cols, categorical_cols


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    return pipeline


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    print("\n=== Kết quả đánh giá mô hình ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="binary", zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average="binary", zero_division=0))
    print("F1 score:", f1_score(y_test, y_pred, average="binary", zero_division=0))
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def plot_churn_distribution(df: pd.DataFrame, target_col: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title("Phân phối churn")
    plt.tight_layout()
    plt.savefig("churn_distribution.png")
    plt.close()


def plot_numeric_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Mối tương quan giữa các thuộc tính số")
    plt.tight_layout()
    plt.savefig("numeric_correlation_matrix.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phân tích hành vi khách hàng và dự đoán churn cho dữ liệu e-commerce."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/ecommerce_customer_churn.csv"),
        help="Đường dẫn tới file CSV dữ liệu e-commerce churn",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("churn_model.pkl"),
        help="File lưu mô hình huấn luyện",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    target_col = find_target_column(df)

    print_basic_info(df, target_col)
    plot_churn_distribution(df, target_col)

    X, y, numeric_cols, categorical_cols = prepare_features(df, target_col)
    plot_numeric_correlations(pd.concat([X[numeric_cols], y], axis=1), numeric_cols)

    if y.nunique() != 2:
        raise ValueError("Chỉ hỗ trợ bài toán phân loại nhị phân. Vui lòng đảm bảo giá trị churn chỉ có 2 lớp.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)
    print("\nMô hình đã huấn luyện xong.")

    evaluate_model(pipeline, X_test, y_test)

    joblib.dump(pipeline, args.output_model)
    print(f"\nMô hình được lưu vào: {args.output_model}")


if __name__ == "__main__":
    main()
