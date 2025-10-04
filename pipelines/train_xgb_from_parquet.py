# pipelines/train_xgb_from_parquet.py
import argparse, pandas as pd
from joblib import load, dump
from xgboost import XGBRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--cleaner", required=True, help="cleaner_xgb.joblib")
    ap.add_argument("--out-model", default="xgb_model.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cleaner = load(args.cleaner)  # 训练/验证时最好直接 transform 原始数据，保持 dtype
    X = pd.read_parquet(args.train_parquet)  # 已是 category dtype
    y = pd.read_csv(args.y)["y"].values

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=args.seed,
        tree_method="hist",
        enable_categorical=True,  # 关键！
        n_jobs=-1
    )
    model.fit(X, y)
    dump(model, args.out_model)
    print("[OK] Saved:", args.out_model)

if __name__ == "__main__":
    main()
