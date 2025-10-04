# pipelines/predict_xgb_from_parquet.py
import argparse, pandas as pd
from joblib import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--x-submit-parquet", required=True)
    ap.add_argument("--test-raw", required=True)  # 为了取 Id
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = load(args.model)
    Xs = pd.read_parquet(args.x_submit_parquet)  # 保留了 category dtype
    test_df = pd.read_csv(args.test_raw)
    pred = model.predict(Xs)
    sub = pd.DataFrame({"Id": test_df["Id"], "SalePrice": pred})
    sub.to_csv(args.out, index=False)
    print("[OK] Saved:", args.out)

if __name__ == "__main__":
    main()
