# pipelines/predict_rf_from_clean.py
# -*- coding: utf-8 -*-
"""
使用已训练的 RandomForest (Raw target) 模型，对清洗后的测试特征 X_submit_clean.csv 进行预测，
并生成 Kaggle 格式的 submission.csv（列：Id, SalePrice）。

特点：
- 读取 feature_names.txt 并严格按该顺序对齐特征列
- 若缺少列 -> 自动补 0；若多出列 -> 自动丢弃
- 仅适用于 Raw 目标模型（不做 log/expm1 转换）
"""

import os
import argparse
import pandas as pd
import numpy as np
from joblib import load


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(p) else None


def read_feature_names(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def align_columns(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """将 df 按 feature_names 对齐：缺失列补 0，多余列丢弃，最终列顺序严格一致。"""
    df_aligned = df.copy()

    # 补缺失列
    missing = [c for c in feature_names if c not in df_aligned.columns]
    for c in missing:
        df_aligned[c] = 0.0

    # 丢弃额外列
    extra = [c for c in df_aligned.columns if c not in feature_names]
    if extra:
        df_aligned = df_aligned.drop(columns=extra)

    # 重排顺序
    df_aligned = df_aligned[feature_names]
    return df_aligned


def main():
    ap = argparse.ArgumentParser(description="Predict with trained RF (Raw target) on cleaned test features.")
    ap.add_argument("--model", required=True, help="Path: trained model.joblib")
    ap.add_argument("--x-submit", required=True, help="Path: X_submit_clean.csv (cleaned test features)")
    ap.add_argument("--test", required=True, help="Path: raw test.csv (to read Id column)")
    ap.add_argument("--feat-names", required=True, help="Path: feature_names.txt from cleaning step")
    ap.add_argument("--out", required=True, help="Path: output submission.csv")
    args = ap.parse_args()

    print("[INFO] Loading model...")
    model = load(args.model)

    print("[INFO] Reading cleaned test features...")
    X_submit = pd.read_csv(args.x_submit)

    print("[INFO] Reading raw test.csv for Id column...")
    df_test_raw = pd.read_csv(args.test)
    if "Id" not in df_test_raw.columns:
        raise ValueError("Raw test.csv must contain an 'Id' column.")

    print("[INFO] Loading feature names and aligning columns...")
    feature_names = read_feature_names(args.feat_names)
    X_submit = align_columns(X_submit, feature_names)

    # 预测（Raw 模型，直接预测，不做 expm1）
    print("[INFO] Predicting...")
    preds = model.predict(X_submit.values)

    # 生成提交文件
    sub = pd.DataFrame({
        "Id": df_test_raw["Id"].values,
        "SalePrice": preds
    })

    ensure_dir(args.out)
    sub.to_csv(args.out, index=False)
    print(f"[OK] Submission saved to: {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
