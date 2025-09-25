from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# ----------------------------- Utils ----------------------------- #
def cap_outliers_iqr(s: pd.Series, whisker: float = 1.5) -> pd.Series:
    """Cap outliers bằng IQR: [Q1 - k*IQR, Q3 + k*IQR]."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - whisker * iqr
    high = q3 + whisker * iqr
    return s.clip(lower=low, upper=high)


def choose_scaler(name: str):
    name = (name or "standard").lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name in ("none", "no", "off", "false"):
        return "passthrough"
    raise ValueError(f"--scale phải là one of: standard|minmax|none, nhận: {name}")


@dataclass
class PrepArtifacts:
    preprocessor_path: str | None
    train_csv: str | None
    test_csv: str | None
    feature_names_path: str | None
    meta_json_path: str | None


# --------------------------- Core logic -------------------------- #
def prepare_dataset(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: str = "standard",
    cap_outliers: bool = True,
    use_smote: bool = False,
    output_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, List[str], PrepArtifacts]:
    """
    Return:
        X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names, artifacts_paths
    """
    # 1) Load
    df = pd.read_csv(csv_path)

    # 2) Basic schema
    target_col = "stroke"
    drop_cols = ["id"]  # id không dùng để train
    numeric_cols = ["age", "avg_glucose_level", "bmi"]
    binary_cols = ["hypertension", "heart_disease"]
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    # Safety checks
    for col in [target_col] + numeric_cols + binary_cols + categorical_cols + drop_cols:
        if col not in df.columns and col not in drop_cols:
            raise KeyError(f"Missing col: {col}")

    # 3) Light cleaning trước khi vào sklearn pipeline
    df_clean = df.copy()

    # Impute quick cho bmi để có thể cap outliers
    if df_clean["bmi"].isna().any():
        df_clean["bmi"] = df_clean["bmi"].fillna(df_clean["bmi"].median())

    # Cap outliers
    if cap_outliers:
        for col in ["bmi", "avg_glucose_level"]:
            df_clean[col] = cap_outliers_iqr(df_clean[col])

    # 4) Split
    y = df_clean[target_col].astype(int)
    X = df_clean.drop(columns=[target_col] + [c for c in drop_cols if c in df_clean.columns], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    # 5) Build preprocessing
    scaler = choose_scaler(scale)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
            ("bin", "passthrough", binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit chỉ trên train để tránh leakage
    preprocessor.fit(X_train)

    # Transform
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Compose feature names
    num_feats = numeric_cols
    cat_ohe = preprocessor.named_transformers_["cat"]["onehot"]
    cat_feats = cat_ohe.get_feature_names_out(categorical_cols).tolist()
    bin_feats = binary_cols
    feature_names = num_feats + cat_feats + bin_feats

    X_train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    # 6) (Optional) SMOTE chỉ trên train
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
        except Exception as e:
            raise RuntimeError("Error") from e

        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train_df.values, y_train.values)
        X_train_df = pd.DataFrame(X_res, columns=feature_names)
        y_train = pd.Series(y_res, name=target_col)

    # 7) (Optional) Lưu artifacts
    artifacts = PrepArtifacts(None, None, None, None, None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save preprocessor
        pre_path = os.path.join(output_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, pre_path)

        # Save transformed CSVs
        train_csv = os.path.join(output_dir, "train_preprocessed.csv")
        test_csv = os.path.join(output_dir, "test_preprocessed.csv")
        tmp_train = X_train_df.copy()
        tmp_train[target_col] = y_train.values
        tmp_test = X_test_df.copy()
        tmp_test[target_col] = y_test.values
        tmp_train.to_csv(train_csv, index=False)
        tmp_test.to_csv(test_csv, index=False)

        # Save feature names
        feat_path = os.path.join(output_dir, "feature_names.txt")
        with open(feat_path, "w", encoding="utf-8") as f:
            for fn in feature_names:
                f.write(f"{fn}\n")

        # Save meta
        meta_path = os.path.join(output_dir, "prep_meta.json")
        meta = {
            "input_csv": os.path.abspath(csv_path),
            "output_dir": os.path.abspath(output_dir),
            "test_size": test_size,
            "random_state": random_state,
            "scale": scale,
            "cap_outliers": cap_outliers,
            "smote": use_smote,
            "n_train": int(len(X_train_df)),
            "n_test": int(len(X_test_df)),
            "pos_rate_train": float(pd.Series(y_train).mean()),
            "pos_rate_test": float(pd.Series(y_test).mean()),
            "n_features": len(feature_names),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        artifacts = PrepArtifacts(
            preprocessor_path=pre_path,
            train_csv=train_csv,
            test_csv=test_csv,
            feature_names_path=feat_path,
            meta_json_path=meta_path,
        )

    return X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names, artifacts


# ------------------------ CLI entrypoint ------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Prepare Kaggle Stroke Prediction dataset")
    p.add_argument("--input", required=True, help="data path (healthcare-dataset-stroke-data.csv)")
    p.add_argument("--output-dir", default=None, help="output dir to save artifacts (preprocessor, transformed CSVs, meta).") 
    p.add_argument("--test-size", type=float, default=0.2, help="test size (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--scale", default="standard", choices=["standard", "minmax", "none"], help="numeric scaling method")
    p.add_argument("--cap-outliers", action="store_true", help="enable IQR outlier capping for bmi & avg_glucose_level")
    p.add_argument("--smote", action="store_true", help="enable SMOTE on training set (requires imblearn)")
    return p.parse_args()


def main():
    args = parse_args()
    X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names, artifacts = prepare_dataset(
        csv_path=args.input,
        test_size=args.test_size,
        random_state=args.random_state,
        scale=args.scale,
        cap_outliers=args.cap_outliers,
        use_smote=args.smote,
        output_dir=args.output_dir,
    )

    # In summary ngắn gọn
    print("=== PREP SUMMARY ===")
    print(f"Train size: {len(X_train_df):,} | Test size: {len(X_test_df):,}")
    print(f"Train pos rate: {float(pd.Series(y_train).mean()):.4f} | Test pos rate: {float(pd.Series(y_test).mean()):.4f}")
    print(f"#Features: {len(feature_names)}")
    if args.output_dir:
        print("Artifacts saved to:")
        print(f"  - Preprocessor: {artifacts.preprocessor_path}")
        print(f"  - Train CSV:    {artifacts.train_csv}")
        print(f"  - Test CSV:     {artifacts.test_csv}")
        print(f"  - Features:     {artifacts.feature_names_path}")
        print(f"  - Meta:         {artifacts.meta_json_path}")


if __name__ == "__main__":
    main()
