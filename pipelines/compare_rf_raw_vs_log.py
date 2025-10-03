# compare_rf_raw_vs_log.py
# -*- coding: utf-8 -*-
"""
对比 RandomForest 在 Raw 目标 vs Log 目标 两种训练方式的效果。
输入：X_clean.csv, y.csv
输出：
- metrics_compare.json
- val_predictions_compare.csv
- compare_residuals.png
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_data(x_path: str, y_path: str):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)["y"].values
    return X.values, y, list(X.columns)


def train_rf(X_train, y_train, n_estimators=800, max_depth=None, random_state=42, n_jobs=-1):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)
    return rf


def eval_model(model, X_val, y_val, log_target=False):
    """返回预测和评估指标"""
    if log_target:
        pred_log = model.predict(X_val)
        pred = np.expm1(pred_log)
    else:
        pred = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    r2 = float(r2_score(y_val, pred))
    return pred, {"rmse": rmse, "r2": r2, "log_target": log_target}


def main():
    parser = argparse.ArgumentParser(description="Compare RF Raw vs Log target")
    parser.add_argument("--x", required=True, help="Path to X_clean.csv")
    parser.add_argument("--y", required=True, help="Path to y.csv")
    parser.add_argument("--outdir", default="rf_compare_outputs", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=1200, help="RF trees")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # === 加载数据 ===
    X, y, feature_names = load_data(args.x, args.y)

    # === train/val split ===
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    results = {}
    preds_df = pd.DataFrame({"y_true": y_va})

    # === Raw 目标 ===
    print("[INFO] Training RandomForest (Raw target)...")
    model_raw = train_rf(X_tr, y_tr, n_estimators=args.n_estimators, random_state=args.random_state)
    pred_raw, metrics_raw = eval_model(model_raw, X_va, y_va, log_target=False)
    results["raw"] = metrics_raw
    preds_df["y_pred_raw"] = pred_raw

    # === Log 目标 ===
    print("[INFO] Training RandomForest (Log target)...")
    y_tr_log = np.log1p(y_tr)
    model_log = train_rf(X_tr, y_tr_log, n_estimators=args.n_estimators, random_state=args.random_state)
    pred_log, metrics_log = eval_model(model_log, X_va, y_va, log_target=True)
    results["log"] = metrics_log
    preds_df["y_pred_log"] = pred_log

    # === 保存指标 ===
    with open(os.path.join(args.outdir, "metrics_compare.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    preds_df.to_csv(os.path.join(args.outdir, "val_predictions_compare.csv"), index=False)

    print("[RESULT] Raw RMSE: {:.4f}, R2: {:.4f}".format(metrics_raw["rmse"], metrics_raw["r2"]))
    print("[RESULT] Log RMSE: {:.4f}, R2: {:.4f}".format(metrics_log["rmse"], metrics_log["r2"]))

    # === 可视化误差对比 ===
    residuals_raw = y_va - pred_raw
    residuals_log = y_va - preds_df["y_pred_log"]

    plt.figure(figsize=(10, 5))
    plt.hist(residuals_raw, bins=40, alpha=0.6, label="Raw residuals")
    plt.hist(residuals_log, bins=40, alpha=0.6, label="Log residuals")
    plt.axvline(0, color="black", linestyle="--")
    plt.legend()
    plt.title("Residuals Distribution: Raw vs Log Target")
    plt.savefig(os.path.join(args.outdir, "compare_residuals.png"), dpi=150)
    plt.close()

    print(f"[OK] Results saved in {args.outdir}")


if __name__ == "__main__":
    main()
