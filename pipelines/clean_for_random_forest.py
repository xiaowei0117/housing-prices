# clean_only_rf.py
# -*- coding: utf-8 -*-
"""
- --save_pipeline  在 train 上拟合并保存清洗器
- --load_pipeline  save the cleaner, transform for test.csv，guarantee align
"""

import argparse
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load

# ============== missing value dictionary ==============
MISSING_MEANS_NONE = {
    "PoolQC": "NoPool",
    "Alley": "NoAlley",
    "Fence": "NoFence",
    "FireplaceQu": "NoFireplace",
    "GarageType": "NoGarage",
    "GarageFinish": "NoGarage",
    "GarageQual": "NoGarage",
    "GarageCond": "NoGarage",
    "BsmtQual": "NoBasement",
    "BsmtCond": "NoBasement",
    "BsmtExposure": "NoBasement",
    "BsmtFinType1": "NoBasement",
    "BsmtFinType2": "NoBasement",
    "MiscFeature": "None",
    "MasVnrType": "None"
}

# ============== tools: Rare category group ==============
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """group the category which is less than rare_thresh as 'Other'"""
    def __init__(self, rare_thresh: float = 0.02):
        self.rare_thresh = rare_thresh
        self.frequent_maps_: Dict[int, set] = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(str)
        self.frequent_maps_.clear()
        for i, col in enumerate(X.columns):
            vc = X[col].value_counts(normalize=True, dropna=False)
            self.frequent_maps_[i] = set(vc[vc >= self.rare_thresh].index.astype(str))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(str)
        for i, col in enumerate(X.columns):
            freq = self.frequent_maps_.get(i, set())
            X[col] = X[col].where(X[col].isin(freq), other="Other")
        return X.values

# ============== special rules + feature engineer ==============
class SpecialRulesAndFeatures(BaseEstimator, TransformerMixin):
    """
    - MSSubClass to str
    - No garage -> GarageYrBlt=0
    - LotFrontage is filled with median of Neighborhood
    - set cap for area：GrLivArea > cap -> cap
    - feature engineer：TotalSF, AgeSinceBuilt, AgeSinceRemod
    """
    def __init__(self,
                 cap_grlivarea: Optional[float] = 4000,
                 neighborhood_col: str = "Neighborhood",
                 lotfront_col: str = "LotFrontage",
                 mssubclass_col: str = "MSSubClass",
                 garage_type_col: str = "GarageType",
                 garage_year_col: str = "GarageYrBlt",
                 create_features: bool = True):
        self.cap = cap_grlivarea
        self.nei = neighborhood_col
        self.lf = lotfront_col
        self.msc = mssubclass_col
        self.gtype = garage_type_col
        self.gyear = garage_year_col
        self.create_features = create_features
        self.group_median_: Optional[pd.Series] = None

    def fit(self, X, y=None):
        X = X.copy()
        if self.lf in X.columns and self.nei in X.columns:
            self.group_median_ = X.groupby(self.nei)[self.lf].median()
        else:
            self.group_median_ = None
        return self

    def transform(self, X):
        X = X.copy()

        # (1) MSSubClass -> categorial
        if self.msc in X.columns:
            X[self.msc] = X[self.msc].astype(str)

        # (2) no garage -> GarageYrBlt = 0
        if (self.gtype in X.columns) and (self.gyear in X.columns):
            no_garage = X[self.gtype].isna() | (X[self.gtype].astype(str).str.lower().isin(["none", "nan"]))
            X.loc[no_garage, self.gyear] = X.loc[no_garage, self.gyear].fillna(0)

        # (3) LotFrontage：filled with Neighborhood median
        if (self.group_median_ is not None) and (self.lf in X.columns) and (self.nei in X.columns):
            need = X[self.lf].isna()
            X.loc[need, self.lf] = X.loc[need, self.nei].map(self.group_median_)

        # (4) cut area
        if self.cap is not None and "GrLivArea" in X.columns:
            X["GrLivArea"] = np.where(X["GrLivArea"].notna(),
                                      np.minimum(X["GrLivArea"], self.cap),
                                      X["GrLivArea"])

        # (5) feature engineer
        if self.create_features:
            for req in ["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"]:
                if req not in X.columns:
                    X[req] = np.nan
            X["TotalSF"] = X["1stFlrSF"] + X["2ndFlrSF"] + X["TotalBsmtSF"]

            for req in ["YrSold", "YearBuilt", "YearRemodAdd"]:
                if req not in X.columns:
                    X[req] = np.nan
            X["AgeSinceBuilt"] = X["YrSold"] - X["YearBuilt"]
            X["AgeSinceRemod"] = X["YrSold"] - X["YearRemodAdd"]

        return X

# ============== build the cleaner ==============
def build_rf_cleaner(df: pd.DataFrame,
                     target: str = "SalePrice",
                     rare_thresh: float = 0.02,
                     cap_grlivarea: Optional[float] = 4000,
                     create_features: bool = True) -> Tuple[Pipeline, List[str]]:
    """return (pipeline, feature_cols), no split and model"""
    cat_none_cols = list(MISSING_MEANS_NONE.keys())

    likely_cat_cols = [
        "MSSubClass","MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope",
        "Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl",
        "Exterior1st","Exterior2nd","ExterQual","ExterCond","Foundation","Heating","HeatingQC",
        "CentralAir","Electrical","KitchenQual","Functional","PavedDrive","SaleType","SaleCondition"
    ]

    likely_num_cols = [
        "LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd",
        "MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF",
        "LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath",
        "BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageYrBlt",
        "GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch",
        "ScreenPorch","PoolArea","MiscVal","MoSold","YrSold",
        "TotalSF","AgeSinceBuilt","AgeSinceRemod"
    ]

    all_features = [c for c in df.columns if c != target and c.lower() != "id"]

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = list(dict.fromkeys([c for c in likely_cat_cols if c in all_features] +
                                  [c for c in obj_cols if c in all_features]))

    cat_none_used = [c for c in cat_none_cols if c in cat_cols]
    cat_other = [c for c in cat_cols if c not in cat_none_used]

    num_cols = [c for c in likely_num_cols if c in all_features]
    auto_num = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    for c in auto_num:
        if (c not in num_cols) and (c in all_features) and (c not in cat_cols):
            num_cols.append(c)

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])

    cat_none_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="None")),
        ("rare", RareCategoryGrouper(rare_thresh=rare_thresh)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    cat_other_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("rare", RareCategoryGrouper(rare_thresh=rare_thresh)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat_none", cat_none_pipe, cat_none_used),
            ("cat_other", cat_other_pipe, cat_other),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("special", SpecialRulesAndFeatures(
            cap_grlivarea=cap_grlivarea,
            create_features=create_features
        )),
        ("columns", ct)
    ])

    return pipeline, all_features

# ============== import the feature names ==============
def extract_feature_names(pipeline: Pipeline) -> List[str]:
    ct = pipeline.named_steps["columns"]
    names: List[str] = []
    # numeral columns
    num_cols = ct.transformers_[0][2]
    names += list(num_cols)
    # cat_none
    cat_none_cols = ct.transformers_[1][2]
    ohe_none = ct.named_transformers_["cat_none"].named_steps["ohe"]
    names += list(ohe_none.get_feature_names_out(cat_none_cols))
    # cat_other
    cat_other_cols = ct.transformers_[2][2]
    ohe_other = ct.named_transformers_["cat_other"].named_steps["ohe"]
    names += list(ohe_other.get_feature_names_out(cat_other_cols))
    return names

# ============== save the output ==============
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_clean_outputs(X_clean: np.ndarray, feature_names: List[str], outdir: str,
                       y: Optional[np.ndarray] = None, save_y: bool = True,
                       filename: str = "X_clean.csv"):
    ensure_dir(outdir)
    fn_path = os.path.join(outdir, "feature_names.txt")
    with open(fn_path, "w", encoding="utf-8") as f:
        for n in feature_names:
            f.write(str(n) + "\n")
    x_path = os.path.join(outdir, filename)
    pd.DataFrame(X_clean, columns=feature_names).to_csv(x_path, index=False)
    print(f"[OK] Saved:\n - {fn_path}\n - {x_path}")
    if save_y and (y is not None):
        y_path = os.path.join(outdir, "y.csv")
        pd.DataFrame({"y": y}).to_csv(y_path, index=False)
        print(f" - {y_path}")

# ============== CLI ==============
def main():
    ap = argparse.ArgumentParser(description="Clean data for RandomForest (no training; just output cleaned matrix).")
    ap.add_argument("--input", required=True, help="Path to input CSV (train.csv or test.csv)")
    ap.add_argument("--outdir", default="rf_clean_outputs", help="Output directory")
    ap.add_argument("--target", default="SalePrice", help="Target column name; use 'None' if absent")
    ap.add_argument("--rare-thresh", type=float, default=0.02, help="Rare category threshold (default 0.02)")
    ap.add_argument("--cap-grlivarea", default="4000", help="Cap for GrLivArea; set 'None' to disable")
    ap.add_argument("--no-create-features", action="store_true", help="Disable engineered features (TotalSF, ages)")
    ap.add_argument("--no-save-y", action="store_true", help="Do not save y.csv even if target exists")
    # 新增：保存/加载已拟合清洗器
    ap.add_argument("--save_pipeline", default=None, help="Path to save fitted cleaner (joblib)")
    ap.add_argument("--load_pipeline", default=None, help="Path to load fitted cleaner (joblib)")
    # 可自定义输出文件名（默认 train: X_clean.csv；test: X_submit_clean.csv）
    ap.add_argument("--output-name", default=None, help="Override output filename (e.g., X_submit_clean.csv)")
    ap.add_argument("--preview-rows", type=int, default=5, help="Preview first N rows of cleaned matrix")
    args = ap.parse_args()

    # analyse cap
    cap_val: Optional[float]
    cap_val = None if str(args.cap_grlivarea).lower() == "none" else float(args.cap_grlivarea)

    # read the data
    df = pd.read_csv(args.input)
    has_target = (args.target.lower() != "none") and (args.target in df.columns)

    # ========== 2 models ==========
    if args.load_pipeline:
        # ---------- have cleaner： transform for test.csv ----------
        cleaner: Pipeline = load(args.load_pipeline)
        X_clean = cleaner.transform(df.drop(columns=[args.target], errors="ignore"))
        feat_names = extract_feature_names(cleaner)

        # output the file name：default X_submit_clean.csv（can be covered with --output-name ）
        out_name = args.output_name or "X_submit_clean.csv"
        save_clean_outputs(X_clean, feat_names, args.outdir,
                           y=None, save_y=False, filename=out_name)

    else:
        # ---------- no cleaner：use the current data + transform（for train.csv） ----------
        pipeline, feature_cols = build_rf_cleaner(
            df,
            target=args.target if has_target else "SalePrice",
            rare_thresh=args.rare_thresh,
            cap_grlivarea=cap_val,
            create_features=(not args.no_create_features)
        )

        X = df[feature_cols].copy()
        y = df[args.target].values if has_target else None

        pipeline.fit(X, y)
        X_clean = pipeline.transform(X)
        feat_names = extract_feature_names(pipeline)

        # train data is saved as X_clean.csv
        out_name = args.output_name or "X_clean.csv"
        save_clean_outputs(X_clean, feat_names, args.outdir,
                           y=y, save_y=(has_target and (not args.no_save_y)),
                           filename=out_name)

        # if Dir is exit，save the cleaner for test.csv
        if args.save_pipeline:
            os.makedirs(os.path.dirname(args.save_pipeline), exist_ok=True)
            dump(pipeline, args.save_pipeline)
            print(f"[OK] Cleaner saved to: {args.save_pipeline}")

    # review
    print("\n[Preview]")
    print(pd.DataFrame(X_clean, columns=feat_names).head(args.preview_rows))

if __name__ == "__main__":
    main()
