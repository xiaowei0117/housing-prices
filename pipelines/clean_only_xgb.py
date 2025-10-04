# pipelines/clean_only_xgb.py
# -*- coding: utf-8 -*-
"""
Cleaner for XGBoost:
- 不做 One-Hot；把分类列转为 pandas.Categorical（保留缺失为 NaN，或把“缺失=无”映射为显式层级）
- 稀有类别聚合为 'Other'（按训练分布拟合，推理时未见过的类别也映射到 'Other'）
- 数值列默认不填充（XGBoost 原生处理 NaN）；可选 --impute-numeric 开启中位数填充
- 特征工程：TotalSF、AgeSinceBuilt、AgeSinceRemod；GrLivArea 可选截断

输出：
- 训练：X_clean.parquet（保留 category dtype）、y.csv、feature_names.txt、cleaner_xgb.joblib
- 测试：X_submit_clean.parquet（保留 category dtype），可选 CSV 浏览

用法：
# 在 train 上拟合并保存 cleaner
python pipelines/clean_only_xgb.py \
  --input data/train.csv \
  --outdir xgb_clean_outputs \
  --save-pipeline xgb_clean_outputs/cleaner_xgb.joblib

# 用已拟合 cleaner 清洗 test，类别自动对齐
python pipelines/clean_only_xgb.py \
  --input data/test.csv \
  --outdir xgb_clean_outputs \
  --load-pipeline xgb_clean_outputs/cleaner_xgb.joblib \
  --target None \
  --output-name X_submit_clean.parquet
"""

import argparse
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin


# ===== 配置：这些列的缺失有业务含义（=没有该设施），我们映射为显式层级 =====
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
    "MasVnrType": "None",
}

# ===== 稀有类别合并（按训练分布） =====
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """将每个分类列中占比 < rare_thresh 的类别合并为 'Other'（fit 时统计，transform 时复用）"""
    def __init__(self, rare_thresh: float = 0.02):
        self.rare_thresh = rare_thresh
        self.frequent_levels_: Dict[str, set] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.frequent_levels_.clear()
        for col in X.columns:
            # 允许缺失；用字符串视图统计（NaN 作为缺失保留，不计入某个水平）
            vc = X[col].astype("string").fillna(pd.NA).value_counts(normalize=True, dropna=True)
            self.frequent_levels_[col] = set(vc[vc >= self.rare_thresh].index.tolist())
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            good = self.frequent_levels_.get(col, set())
            # 仅在值非缺失时做合并；缺失维持 NaN，让 XGB 自处理
            mask = X[col].notna()
            X.loc[mask, col] = X.loc[mask, col].astype("string").where(
                X.loc[mask, col].astype("string").isin(good), other="Other"
            )
        return X

# ===== 特殊规则 + 特征工程 =====
class SpecialRulesAndFeatures(BaseEstimator, TransformerMixin):
    """
    - MSSubClass → 字符串（类别）
    - 无车库 -> GarageYrBlt=0
    - LotFrontage：按 Neighborhood 的中位数填补（其余保留 NaN）
    - GrLivArea 可选上限截断
    - 工程特征：TotalSF, AgeSinceBuilt, AgeSinceRemod
    """
    def __init__(self,
                 cap_grlivarea: Optional[float] = 4000.0,
                 create_features: bool = True):
        self.cap = cap_grlivarea
        self.group_median_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y=None):
        if {"LotFrontage", "Neighborhood"}.issubset(X.columns):
            self.group_median_ = X.groupby("Neighborhood")["LotFrontage"].median()
        else:
            self.group_median_ = None
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # (1) MSSubClass → 类别（字符串）
        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype("Int64").astype("string")

        # (2) 无车库 -> GarageYrBlt = 0
        if "GarageType" in X.columns and "GarageYrBlt" in X.columns:
            no_garage = X["GarageType"].isna() | (X["GarageType"].astype("string").str.lower().isin(["nan", "none"]))
            X.loc[no_garage, "GarageYrBlt"] = X.loc[no_garage, "GarageYrBlt"].fillna(0)

        # (3) LotFrontage：按 Neighborhood 中位数填补（其余缺失保留）
        if self.group_median_ is not None and "LotFrontage" in X.columns and "Neighborhood" in X.columns:
            need = X["LotFrontage"].isna()
            X.loc[need, "LotFrontage"] = X.loc[need, "Neighborhood"].map(self.group_median_)

        # (4) GrLivArea 截断
        if self.cap is not None and "GrLivArea" in X.columns:
            X["GrLivArea"] = np.where(
                X["GrLivArea"].notna(), np.minimum(X["GrLivArea"], self.cap), X["GrLivArea"]
            )

        # (5) 工程特征
        if self.create_features:
            for req in ["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"]:
                if req not in X.columns: X[req] = np.nan
            X["TotalSF"] = X["1stFlrSF"] + X["2ndFlrSF"] + X["TotalBsmtSF"]

            for req in ["YrSold", "YearBuilt", "YearRemodAdd"]:
                if req not in X.columns: X[req] = np.nan
            X["AgeSinceBuilt"] = X["YrSold"] - X["YearBuilt"]
            X["AgeSinceRemod"] = X["YrSold"] - X["YearRemodAdd"]

        return X

# ===== XGB 清洗器（不做 One-Hot；输出 DataFrame 且分类列为 category dtype） =====
class XGBCleaner(BaseEstimator, TransformerMixin):
    def __init__(self,
                 rare_thresh: float = 0.02,
                 cap_grlivarea: Optional[float] = 4000.0,
                 create_features: bool = True,
                 impute_numeric: bool = False):
        self.rare_thresh = rare_thresh
        self.cap = cap_grlivarea
        self.create_features = create_features
        self.impute_numeric = impute_numeric

        self.special_ = SpecialRulesAndFeatures(cap_grlivarea=self.cap, create_features=self.create_features)
        self.rare_ = RareCategoryGrouper(rare_thresh=self.rare_thresh)

        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.cat_levels_: Dict[str, List[str]] = {}

    def _detect_cols(self, X: pd.DataFrame, target: Optional[str]) -> List[str]:
        all_features = [c for c in X.columns if (target is None or c != target) and (c.lower() != "id")]
        # 预选类别列（包含 object + 一些业务列）
        likely_cat = [
            "MSSubClass","MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope",
            "Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl",
            "Exterior1st","Exterior2nd","ExterQual","ExterCond","Foundation","Heating","HeatingQC",
            "CentralAir","Electrical","KitchenQual","Functional","PavedDrive","SaleType","SaleCondition"
        ]
        obj_cols = X[all_features].select_dtypes(include=["object"]).columns.tolist()
        cat = list(dict.fromkeys([c for c in likely_cat if c in all_features] + obj_cols))

        # “缺失=无”的列并入类别列
        cat_none_cols = [c for c in MISSING_MEANS_NONE.keys() if c in all_features]
        cat = list(dict.fromkeys(cat + cat_none_cols))

        # 数值列：按 dtype 推断，且不包含已判定的类别列
        num = [c for c in X[all_features].select_dtypes(include=["number"]).columns if c not in cat]

        self.cat_cols_ = cat
        self.num_cols_ = num
        return all_features

    def fit(self, X: pd.DataFrame, y=None):
        Xw = self.special_.fit_transform(X)

        # 1) 缺失=无 的列：填充为显式层级
        for c, lvl in MISSING_MEANS_NONE.items():
            if c in Xw.columns:
                Xw[c] = Xw[c].astype("string")
                Xw[c] = Xw[c].fillna(lvl).replace({"nan": lvl, "None": lvl})

        # 列检测
        self._detect_cols(Xw, target=None)

        # 2) 稀有类别合并（fit on train）
        cat_df = Xw[self.cat_cols_].copy()
        cat_df = self.rare_.fit_transform(cat_df)

        # 3) 记录每一列的类别集合（含 'Other'，缺失不计入）
        self.cat_levels_.clear()
        for c in cat_df.columns:
            levels = pd.Series(cat_df[c].dropna().astype("string").unique()).tolist()
            if "Other" not in levels:
                levels.append("Other")
            self.cat_levels_[c] = sorted([str(v) for v in levels])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xw = self.special_.transform(X)

        # 缺失=无 的列：显式层级
        for c, lvl in MISSING_MEANS_NONE.items():
            if c in Xw.columns:
                Xw[c] = Xw[c].astype("string")
                Xw[c] = Xw[c].fillna(lvl).replace({"nan": lvl, "None": lvl})

        # 确保列定义
        self._detect_cols(Xw, target=None)

        # 稀有类别映射（基于训练时的 frequent set）
        cat_df = Xw[self.cat_cols_].copy()
        # 先根据训练分布把不在 frequent 内的值映射为 'Other'
        for c in cat_df.columns:
            # 使用 RareCategoryGrouper 的规则：仅非缺失时进行映射
            mask = cat_df[c].notna()
            good = self.rare_.frequent_levels_.get(c, set())
            cat_df.loc[mask, c] = cat_df.loc[mask, c].astype("string").where(
                cat_df.loc[mask, c].astype("string").isin(good), other="Other"
            )

        # 转为 category dtype，并对齐训练时的类别集合（未知类别统一为 'Other'）
        for c in cat_df.columns:
            levels = self.cat_levels_.get(c, ["Other"])
            s = cat_df[c].astype("string")
            s = s.where(s.isin(levels), other="Other")
            cat_df[c] = pd.Categorical(s, categories=levels)

        # 数值列（保留为 float；是否填充由 impute_numeric 控制）
        num_df = Xw[self.num_cols_].copy()
        if self.impute_numeric:
            for c in num_df.columns:
                if num_df[c].isna().any():
                    num_df[c] = num_df[c].fillna(num_df[c].median())

        # 合并回去（保持列顺序：数值在前，类别在后，便于阅读）
        X_out = pd.concat([num_df, cat_df], axis=1)

        return X_out


def ensure_dir(p: str): os.makedirs(p, exist_ok=True)


def save_df(df: pd.DataFrame, outdir: str, base_name: str, also_csv: bool = True):
    ensure_dir(outdir)
    pq = os.path.join(outdir, base_name if base_name.endswith(".parquet") else f"{base_name}.parquet")
    df.to_parquet(pq, index=False)
    print(f"[OK] Saved Parquet (preserves dtypes): {pq}")
    if also_csv:
        csv = os.path.join(outdir, base_name.replace(".parquet", ".csv") if base_name.endswith(".parquet") else f"{base_name.replace('.csv','')}.csv")
        df.to_csv(csv, index=False)
        print(f"[OK] Saved CSV (for preview only, dtypes lost): {csv}")


def main():
    ap = argparse.ArgumentParser(description="Clean data for XGBoost (no one-hot; categorical dtype).")
    ap.add_argument("--input", required=True, help="Path to input CSV (train.csv or test.csv)")
    ap.add_argument("--outdir", default="xgb_clean_outputs", help="Output directory")
    ap.add_argument("--target", default="SalePrice", help="Target column name; use 'None' if absent")
    ap.add_argument("--rare-thresh", type=float, default=0.02, help="Rare category threshold")
    ap.add_argument("--cap-grlivarea", default="4000", help="Cap for GrLivArea; 'None' to disable")
    ap.add_argument("--no-create-features", action="store_true", help="Disable engineered features")
    ap.add_argument("--impute-numeric", action="store_true", help="Median-impute numeric NaNs (default off)")
    ap.add_argument("--save-pipeline", default=None, help="Path to save fitted cleaner (joblib)")
    ap.add_argument("--load-pipeline", default=None, help="Path to load fitted cleaner (joblib)")
    ap.add_argument("--output-name", default=None, help="Override output filename (e.g., X_submit_clean.parquet)")
    ap.add_argument("--preview-rows", type=int, default=5, help="Preview first N rows")
    args = ap.parse_args()

    cap_val: Optional[float] = None if str(args.cap_grlivarea).lower() == "none" else float(args.cap_grlivarea)
    has_target = (args.target.lower() != "none")

    df = pd.read_csv(args.input)

    if args.load_pipeline:
        # ---- 推理阶段：用已拟合 cleaner 直接 transform（test.csv）----
        cleaner: XGBCleaner = load(args.load_pipeline)
        X = df.drop(columns=[args.target], errors="ignore")
        X_clean = cleaner.transform(X)
        out_name = args.output_name or "X_submit_clean.parquet"
        save_df(X_clean, args.outdir, out_name, also_csv=True)

        print("\n[Preview]")
        print(X_clean.head(args.preview_rows))
        return

    # ---- 训练阶段：拟合 cleaner 并保存 ----
    cleaner = XGBCleaner(
        rare_thresh=args.rare_thresh,
        cap_grlivarea=cap_val,
        create_features=(not args.no_create_features),
        impute_numeric=args.impute_numeric
    )

    X = df.drop(columns=[args.target], errors="ignore") if has_target and args.target in df.columns else df.copy()
    y = df[args.target].values if has_target and args.target in df.columns else None

    cleaner.fit(X, y)
    X_clean = cleaner.transform(X)

    # 保存输出
    out_name = args.output_name or "X_clean.parquet"
    save_df(X_clean, args.outdir, out_name, also_csv=True)

    # y 与特征名（仅作参考；训练时建议直接用 parquet + cleaner）
    if y is not None:
        y_path = os.path.join(args.outdir, "y.csv")
        pd.DataFrame({"y": y}).to_csv(y_path, index=False)
        print(f"[OK] Saved y: {y_path}")

    # 导出列名（注意：CSV 会丢 dtype，这里仅提供列顺序参考）
    feat_names = list(X_clean.columns)
    fn_path = os.path.join(args.outdir, "feature_names.txt")
    with open(fn_path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, feat_names)))
    print(f"[OK] Saved feature names: {fn_path}")

    if args.save_pipeline:
        os.makedirs(os.path.dirname(args.save_pipeline), exist_ok=True)
        dump(cleaner, args.save_pipeline)
        print(f"[OK] Cleaner saved to: {args.save_pipeline}")

    print("\n[Preview]")
    print(X_clean.head(args.preview_rows))


if __name__ == "__main__":
    main()
