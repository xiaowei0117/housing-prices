# pipelines/train_rf_from_clean.py
# -*- coding: utf-8 -*-
"""
Train RandomForest on cleaned data (X_clean.csv, y.csv).
- support --log-target（训练用 log1p(y)，validation with original scale，save y_pred & y_pred_log）
- --max-features: support None / sqrt / log2 / 1.0 etc
- output：model.joblib、metrics.json、val_predictions.csv、feature_importance*.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# ------------------ Utils ------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_max_features(val):
    """
     --max-features:
      - None / none / null  -> None
      - 'sqrt' / 'log2'     -> str
      - float/int str     -> float
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip().lower()
    if s in ("none", "null"):
        return None
    if s in ("sqrt", "log2"):
        return s
    try:
        return float(s)  # support 1.0, 0.5 等
    except ValueError:
        raise ValueError(f"Invalid --max-features: {val}. Use None/sqrt/log2 or a float in (0,1].")


def load_data(x_path: str, y_path: str, feat_names_path: str | None = None):
    Xdf = pd.read_csv(x_path)
    y = pd.read_csv(y_path)["y"].values
    feature_names = list(Xdf.columns) if feat_names_path is None else open(
        feat_names_path, "r", encoding="utf-8"
    ).read().splitlines()
    return Xdf.values, y, feature_names


def train_baseline_rf(
    X_train,
    y_train,
    n_estimators=800,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt",
):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        # bootstrap=True  # 默认 True
    )
    rf.fit(X_train, y_train)
    return rf


def random_search_rf(
    X_train,
    y_train,
    random_state=42,
    n_jobs=-1,
    n_iter=25,
):
    param_dist = {
        "n_estimators": [400, 600, 800, 1000, 1200],
        "max_depth": [None, 12, 16, 20, 24, 28, 32],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", None, 1.0],
        "bootstrap": [True],
    }
    base = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_


def eval_and_dump(
    model,
    X_val,
    y_true_original,
    outdir,
    feature_names,
    topk=40,
    log_target=False,
):
    """
    - always validation on original scale(y_true_original 未 log1p)
    - when log_target=True，predict in log space，then expm1
    - save：y_true, y_pred（original scale）, and y_pred_log（if log_target=True）
    """
    pred_raw = model.predict(X_val)                         # if log_target=True，this is log prediction
    pred = np.expm1(pred_raw) if log_target else pred_raw   # back to original scale

    #  sklearn： sqrt(MSE) to RMSE
    mse = mean_squared_error(y_true_original, pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_original, pred))

    ensure_dir(outdir)
    metrics = {"rmse": rmse, "r2": r2, "log_target": bool(log_target)}
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # output prediction
    pred_df = {
        "y_true": y_true_original,
        "y_pred": pred,  # original scale
    }
    if log_target:
        pred_df["y_pred_log"] = pred_raw  # log space prediction
    pd.DataFrame(pred_df).to_csv(os.path.join(outdir, "val_predictions.csv"), index=False)

    # feather importance
    fi = pd.DataFrame(
        {"feature": feature_names, "importance_model": model.feature_importances_}
    ).sort_values("importance_model", ascending=False)
    fi_head = fi.head(topk)
    fi.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
    fi_head.to_csv(os.path.join(outdir, "feature_importance_topk.csv"), index=False)

    return metrics, fi_head


# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description="Train RandomForest on cleaned data (X_clean/y).")
    parser.add_argument("--x", required=True, help="Path to X_clean.csv")
    parser.add_argument("--y", required=True, help="Path to y.csv")
    parser.add_argument("--feat-names", default=None, help="Optional path to feature_names.txt")
    parser.add_argument("--outdir", default="rf_model_outputs", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=800, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max depth")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Min samples per leaf")
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Max features per split: int/float, 'sqrt', 'log2', or None",
    )
    parser.add_argument("--do-search", action="store_true", help="Run RandomizedSearchCV")
    parser.add_argument("--search-iter", type=int, default=25, help="Search iterations")
    parser.add_argument("--topk", type=int, default=40, help="Export top-K feature importances")
    parser.add_argument("--log-target", action="store_true", help="Use log1p(y) for training target")
    args = parser.parse_args()

    #  max_features（support None/sqrt/log2/float）
    args.max_features = parse_max_features(args.max_features)

    ensure_dir(args.outdir)

    # read data
    X, y, feature_names = load_data(args.x, args.y, args.feat_names)

    # split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # target selection log1p
    y_tr_fit = np.log1p(y_tr) if args.log_target else y_tr

    # training
    if args.do_search:
        print("[INFO] Running RandomizedSearchCV...")
        model, best_params = random_search_rf(
            X_tr, y_tr_fit, random_state=args.random_state, n_iter=args.search_iter
        )
        with open(os.path.join(args.outdir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print("[INFO] Best params:", best_params)
    else:
        print("[INFO] Training baseline RandomForest...")
        model = train_baseline_rf(
            X_tr,
            y_tr_fit,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
        )

    # validation and output（always input y_va；log_target control if it's original scale）
    metrics, fi_top = eval_and_dump(
        model, X_va, y_va, args.outdir, feature_names, topk=args.topk, log_target=args.log_target
    )
    print("[RESULT] RMSE: {:.4f}, R2: {:.4f}".format(metrics["rmse"], metrics["r2"]))
    print("[INFO] Top feature importance (head):")
    print(fi_top.head(10))

    # save the model
    model_path = os.path.join(args.outdir, "model.joblib")
    dump(model, model_path)
    print(f"[OK] Model saved to: {model_path}")


if __name__ == "__main__":
    main()
