# import os
# import gc
# import json
# import time
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import ExtraTreesRegressor

# import lightgbm as lgb
# import xgboost as xgb

# # -----------------------
# # PATHS
# # -----------------------
# TRAIN_PATH = "data/train.csv"
# TEST_PATH  = "data/test.csv"
# OUT_DIR    = "preds"                 # where we store OOF/test preds for each run
# SUB_DIR    = "subs"                  # output submissions
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(SUB_DIR, exist_ok=True)

# TARGET = "exam_score"
# ID_COL = "id"

# # -----------------------
# # GLOBAL SETTINGS
# # -----------------------
# N_SPLITS = 5
# LOW_CARD_MAX_UNIQUE = 30

# # Overnight seed budgets (adjust if you want more/less)
# SEEDS_LGBM = [42, 2026, 7, 111, 999, 1234, 888, 314, 2718, 17]      # 10 seeds
# SEEDS_XGB  = [42, 2026, 7, 111, 999]                                # 5 seeds
# SEEDS_ET   = [42, 2026, 7]                                          # 3 seeds

# RANDOM_STATE_STACK = 42
# RIDGE_ALPHA = 1.0

# # -----------------------
# # Utils
# # -----------------------
# def rmse(y_true, y_pred) -> float:
#     return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# def log(msg: str):
#     print(msg, flush=True)

# def save_json(path: str, obj: dict):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2)

# # -----------------------
# # Feature prep
# # -----------------------
# def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
#     y = train_df[TARGET].astype(float)
#     X = train_df.drop(columns=[TARGET]).copy()
#     X_test = test_df.copy()

#     # Categoricals by dtype
#     cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]

#     # Promote low-cardinality integer/bool to categorical (often helps synthetic data)
#     for c in X.columns:
#         if c == ID_COL or c in cat_cols:
#             continue
#         if pd.api.types.is_integer_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
#             if X[c].nunique(dropna=True) <= LOW_CARD_MAX_UNIQUE:
#                 cat_cols.append(c)

#     num_cols = [c for c in X.columns if c not in cat_cols and c != ID_COL]

#     # Fill missing
#     for c in cat_cols:
#         X[c] = X[c].astype("string").fillna("Missing")
#         X_test[c] = X_test[c].astype("string").fillna("Missing")

#     medians = X[num_cols].median(numeric_only=True)
#     X[num_cols] = X[num_cols].fillna(medians)
#     X_test[num_cols] = X_test[num_cols].fillna(medians)

#     # Drop ID
#     feature_cols = [c for c in X.columns if c != ID_COL]
#     X = X[feature_cols]
#     X_test = X_test[feature_cols]

#     return X, y, X_test, cat_cols

# # -----------------------
# # Caching helpers
# # -----------------------
# def pred_paths(tag: str):
#     return (
#         os.path.join(OUT_DIR, f"{tag}_oof.npy"),
#         os.path.join(OUT_DIR, f"{tag}_test.npy"),
#         os.path.join(OUT_DIR, f"{tag}_meta.json"),
#     )

# def already_done(tag: str) -> bool:
#     oof_p, test_p, meta_p = pred_paths(tag)
#     return os.path.exists(oof_p) and os.path.exists(test_p) and os.path.exists(meta_p)

# def save_preds(tag: str, oof: np.ndarray, test_pred: np.ndarray, meta: dict):
#     oof_p, test_p, meta_p = pred_paths(tag)
#     np.save(oof_p, oof)
#     np.save(test_p, test_pred)
#     save_json(meta_p, meta)

# def load_preds(tag: str):
#     oof_p, test_p, meta_p = pred_paths(tag)
#     return np.load(oof_p), np.load(test_p)

# # -----------------------
# # LightGBM trainer
# # -----------------------
# def run_lgbm(X, y, X_test, cat_cols, seed: int, params: dict, tag: str):
#     if already_done(tag):
#         log(f"[LGBM] SKIP {tag} (cached)")
#         return

#     X_l = X.copy()
#     X_test_l = X_test.copy()
#     for c in cat_cols:
#         if c in X_l.columns:
#             X_l[c] = X_l[c].astype("category")
#             X_test_l[c] = X_test_l[c].astype("category")

#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X_l), dtype=float)
#     test_pred = np.zeros(len(X_test_l), dtype=float)

#     t0 = time.time()
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_l), 1):
#         X_tr, X_va = X_l.iloc[tr_idx], X_l.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

#         train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, free_raw_data=False)
#         valid_set = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, free_raw_data=False)

#         model = lgb.train(
#             params,
#             train_set,
#             num_boost_round=20000,
#             valid_sets=[valid_set],
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds=300, verbose=False),
#                 lgb.log_evaluation(period=0),
#             ],
#         )

#         va_pred = model.predict(X_va, num_iteration=model.best_iteration)
#         oof[va_idx] = va_pred
#         test_pred += model.predict(X_test_l, num_iteration=model.best_iteration) / N_SPLITS

#         log(f"[LGBM] {tag} | Fold {fold} RMSE: {rmse(y_va, va_pred):.6f} | best_iter={model.best_iteration}")

#     cv = rmse(y, oof)
#     meta = {"family": "lgbm", "seed": seed, "cv_rmse": cv, "params": params}
#     save_preds(tag, oof, test_pred, meta)

#     log(f"[LGBM] DONE {tag} | CV RMSE: {cv:.6f} | time: {time.time()-t0:.1f}s")
#     gc.collect()

# # -----------------------
# # XGBoost trainer
# # -----------------------
# def run_xgb(X, y, X_test, cat_cols, seed: int, params: dict, tag: str):
#     if already_done(tag):
#         log(f"[XGB] SKIP {tag} (cached)")
#         return

#     # One-hot for XGB (simple, works well)
#     X_all = pd.concat([X, X_test], axis=0).copy()

#     for c in cat_cols:
#         if c in X_all.columns:
#             X_all[c] = X_all[c].astype("string").fillna("Missing")

#     X_all = pd.get_dummies(X_all, columns=[c for c in cat_cols if c in X_all.columns], dummy_na=False)

#     X_xgb = X_all.iloc[: len(X), :]
#     X_test_xgb = X_all.iloc[len(X):, :]

#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X_xgb), dtype=float)
#     test_pred = np.zeros(len(X_test_xgb), dtype=float)

#     t0 = time.time()
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_xgb), 1):
#         X_tr, X_va = X_xgb.iloc[tr_idx], X_xgb.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

#         dtr = xgb.DMatrix(X_tr, label=y_tr)
#         dva = xgb.DMatrix(X_va, label=y_va)
#         dte = xgb.DMatrix(X_test_xgb)

#         booster = xgb.train(
#             params,
#             dtr,
#             num_boost_round=50000,
#             evals=[(dva, "valid")],
#             verbose_eval=False,
#             early_stopping_rounds=500,
#         )

#         va_pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
#         oof[va_idx] = va_pred
#         test_pred += booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)) / N_SPLITS

#         log(f"[XGB] {tag} | Fold {fold} RMSE: {rmse(y_va, va_pred):.6f} | best_iter={booster.best_iteration}")

#     cv = rmse(y, oof)
#     meta = {"family": "xgb", "seed": seed, "cv_rmse": cv, "params": params}
#     save_preds(tag, oof, test_pred, meta)

#     log(f"[XGB] DONE {tag} | CV RMSE: {cv:.6f} | time: {time.time()-t0:.1f}s")
#     gc.collect()

# # -----------------------
# # ExtraTrees trainer (diversity)
# # -----------------------
# def run_et(X, y, X_test, cat_cols, seed: int, tag: str):
#     if already_done(tag):
#         log(f"[ET] SKIP {tag} (cached)")
#         return

#     # One-hot for ET
#     X_all = pd.concat([X, X_test], axis=0).copy()
#     for c in cat_cols:
#         if c in X_all.columns:
#             X_all[c] = X_all[c].astype("string").fillna("Missing")
#     X_all = pd.get_dummies(X_all, columns=[c for c in cat_cols if c in X_all.columns], dummy_na=False)

#     X_et = X_all.iloc[: len(X), :]
#     X_test_et = X_all.iloc[len(X):, :]

#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
#     oof = np.zeros(len(X_et), dtype=float)
#     test_pred = np.zeros(len(X_test_et), dtype=float)

#     model = ExtraTreesRegressor(
#         n_estimators=1200,
#         max_features=0.7,
#         min_samples_leaf=2,
#         random_state=seed,
#         n_jobs=-1,
#     )

#     t0 = time.time()
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_et), 1):
#         X_tr, X_va = X_et.iloc[tr_idx], X_et.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

#         model.fit(X_tr, y_tr)
#         va_pred = model.predict(X_va)
#         oof[va_idx] = va_pred
#         test_pred += model.predict(X_test_et) / N_SPLITS

#         log(f"[ET] {tag} | Fold {fold} RMSE: {rmse(y_va, va_pred):.6f}")

#     cv = rmse(y, oof)
#     meta = {"family": "extratrees", "seed": seed, "cv_rmse": cv}
#     save_preds(tag, oof, test_pred, meta)

#     log(f"[ET] DONE {tag} | CV RMSE: {cv:.6f} | time: {time.time()-t0:.1f}s")
#     gc.collect()

# # -----------------------
# # Stacking + output
# # -----------------------
# def collect_tags():
#     tags = []
#     for fn in os.listdir(OUT_DIR):
#         if fn.endswith("_meta.json"):
#             tags.append(fn.replace("_meta.json", ""))
#     tags = sorted(set(tags))
#     return tags

# def load_meta(tag: str):
#     _, _, meta_p = pred_paths(tag)
#     with open(meta_p, "r", encoding="utf-8") as f:
#         return json.load(f)

# def build_stack_and_write(test_ids):
#     tags = collect_tags()
#     if not tags:
#         raise RuntimeError("No predictions found in preds/. Nothing to stack.")

#     # Load OOF/test preds for each tag
#     oofs = []
#     tests = []
#     metas = []
#     for tag in tags:
#         oof, te = load_preds(tag)
#         oofs.append(oof)
#         tests.append(te)
#         metas.append(load_meta(tag))

#     X_oof = np.vstack(oofs).T
#     X_test = np.vstack(tests).T

#     # Simple mean baseline across all models
#     mean_test = X_test.mean(axis=1)

#     # Ridge stacking (robust)
#     ridge = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE_STACK)
#     ridge.fit(X_oof, y_global)
#     stack_test = ridge.predict(X_test)

#     # Also: OOF-optimized blend of top K models by CV (sometimes cleaner)
#     # Take top 8 runs and do ridge on them (reduces noise)
#     cv_sorted = sorted([(m["cv_rmse"], i) for i, m in enumerate(metas)], key=lambda x: x[0])
#     topk = min(8, len(cv_sorted))
#     idxs = [i for _, i in cv_sorted[:topk]]
#     ridge2 = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE_STACK)
#     ridge2.fit(X_oof[:, idxs], y_global)
#     stack_topk_test = ridge2.predict(X_test[:, idxs])

#     # Write submissions
#     def write_sub(filename, pred):
#         df = pd.DataFrame({ID_COL: test_ids, TARGET: pred})
#         df.to_csv(os.path.join(SUB_DIR, filename), index=False)
#         log(f"Saved {os.path.join(SUB_DIR, filename)} ✅")

#     write_sub("submission_mean.csv", mean_test)
#     write_sub("submission_stack.csv", stack_test)
#     write_sub("submission_stack_topk.csv", stack_topk_test)

#     # Print summary
#     log("\nTop runs by CV:")
#     for rank, (cv, i) in enumerate(cv_sorted[:10], 1):
#         log(f"{rank:2d}. {tags[i]} | CV {cv:.6f} | {metas[i]['family']} seed={metas[i].get('seed')}")
#     log("\nStacking complete.")

# # -----------------------
# # MAIN
# # -----------------------
# train_df = pd.read_csv(TRAIN_PATH)
# test_df = pd.read_csv(TEST_PATH)

# X_global, y_global, X_test_global, cat_cols_global = prepare_features(train_df, test_df)
# test_ids = test_df[ID_COL].values

# log(f"Train: {X_global.shape}  Test: {X_test_global.shape}  Cat cols: {len(cat_cols_global)}")
# log(f"Output dirs: {OUT_DIR}/ and {SUB_DIR}/")

# # ---- LGBM param sets (2 variants)
# LGBM_PARAMS_A = dict(
#     objective="regression",
#     metric="rmse",
#     learning_rate=0.03,
#     num_leaves=256,
#     max_depth=-1,
#     min_data_in_leaf=50,
#     feature_fraction=0.80,
#     bagging_fraction=0.80,
#     bagging_freq=1,
#     lambda_l2=5.0,
#     verbose=-1,
# )

# # More capacity + more reg (often helps)
# LGBM_PARAMS_B = dict(
#     objective="regression",
#     metric="rmse",
#     learning_rate=0.03,
#     num_leaves=512,
#     max_depth=-1,
#     min_data_in_leaf=80,
#     feature_fraction=0.75,
#     bagging_fraction=0.75,
#     bagging_freq=1,
#     lambda_l2=8.0,
#     verbose=-1,
# )

# # ---- XGB params (good default)
# XGB_PARAMS = dict(
#     objective="reg:squarederror",
#     eval_metric="rmse",
#     eta=0.03,
#     max_depth=10,
#     min_child_weight=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_lambda=8.0,
#     reg_alpha=0.0,
#     tree_method="hist",   # fast CPU; switch to "gpu_hist" if you want GPU + CUDA works well
#     seed=0,               # will override per run
# )

# # -----------------------
# # Train library
# # -----------------------
# # LightGBM A
# for s in SEEDS_LGBM:
#     tag = f"lgbm_A_seed{s}"
#     run_lgbm(X_global, y_global, X_test_global, cat_cols_global, s, LGBM_PARAMS_A, tag)

# # LightGBM B
# for s in SEEDS_LGBM:
#     tag = f"lgbm_B_seed{s}"
#     run_lgbm(X_global, y_global, X_test_global, cat_cols_global, s, LGBM_PARAMS_B, tag)

# # XGBoost
# for s in SEEDS_XGB:
#     params = dict(XGB_PARAMS)
#     params["seed"] = s
#     tag = f"xgb_seed{s}"
#     run_xgb(X_global, y_global, X_test_global, cat_cols_global, s, params, tag)

# # ExtraTrees diversity
# for s in SEEDS_ET:
#     tag = f"et_seed{s}"
#     run_et(X_global, y_global, X_test_global, cat_cols_global, s, tag)

# # -----------------------
# # Stack + write submissions
# # -----------------------
# build_stack_and_write(test_ids)


# _________________________________________________________________________________________________________________
import os
import gc
import json
import time
import math
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge

import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# PATHS
# =========================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"

OUT_DIR = "preds"   # cached preds per run
SUB_DIR = "subs"    # submissions
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = "exam_score"
ID_COL = "id"

# =========================
# GLOBAL SETTINGS
# =========================
N_SPLITS = 5
LOW_CARD_MAX_UNIQUE = 30

# Overnight budgets (tune as needed)
SEEDS_LGBM = [42, 2026, 7, 111, 999, 1234, 888, 314, 2718, 17]   # 10
SEEDS_XGB  = [42, 2026, 7, 111, 999]                             # 5
SEEDS_NN   = [42, 2026, 7]                                       # 3 (NN is expensive; add more later)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# Meta models
META_MODELS = {
    "ridge": Ridge(alpha=1.0, random_state=42),
    "enet":  ElasticNet(alpha=1e-4, l1_ratio=0.1, random_state=42, max_iter=20000),
    "bayes": BayesianRidge()
}

# =========================
# UTIL
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def log(msg: str):
    print(msg, flush=True)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# =========================
# FEATURE PREP
# =========================
def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    y = train_df[TARGET].astype(float)
    X = train_df.drop(columns=[TARGET]).copy()
    X_test = test_df.copy()

    # categoricals by dtype
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]

    # promote low-cardinality integer/bool to categorical (synthetic tabular win)
    for c in X.columns:
        if c == ID_COL or c in cat_cols:
            continue
        if pd.api.types.is_integer_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            if X[c].nunique(dropna=True) <= LOW_CARD_MAX_UNIQUE:
                cat_cols.append(c)

    num_cols = [c for c in X.columns if c not in cat_cols and c != ID_COL]

    # fill missing
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Missing")
        X_test[c] = X_test[c].astype("string").fillna("Missing")

    medians = X[num_cols].median(numeric_only=True)
    X[num_cols] = X[num_cols].fillna(medians)
    X_test[num_cols] = X_test[num_cols].fillna(medians)

    # drop id
    feat_cols = [c for c in X.columns if c != ID_COL]
    X = X[feat_cols]
    X_test = X_test[feat_cols]

    return X, y, X_test, cat_cols, num_cols

# =========================
# CACHING
# =========================
def pred_paths(tag: str):
    return (
        os.path.join(OUT_DIR, f"{tag}_oof.npy"),
        os.path.join(OUT_DIR, f"{tag}_test.npy"),
        os.path.join(OUT_DIR, f"{tag}_meta.json"),
    )

def already_done(tag: str) -> bool:
    oof_p, test_p, meta_p = pred_paths(tag)
    return os.path.exists(oof_p) and os.path.exists(test_p) and os.path.exists(meta_p)

def save_preds(tag: str, oof: np.ndarray, test_pred: np.ndarray, meta: dict):
    oof_p, test_p, meta_p = pred_paths(tag)
    np.save(oof_p, oof)
    np.save(test_p, test_pred)
    save_json(meta_p, meta)

def load_preds(tag: str):
    oof_p, test_p, _ = pred_paths(tag)
    return np.load(oof_p), np.load(test_p)

def load_meta(tag: str):
    _, _, meta_p = pred_paths(tag)
    with open(meta_p, "r", encoding="utf-8") as f:
        return json.load(f)

def list_tags():
    tags = []
    for fn in os.listdir(OUT_DIR):
        if fn.endswith("_meta.json"):
            tags.append(fn.replace("_meta.json", ""))
    return sorted(set(tags))

# =========================
# LIGHTGBM RUNS
# =========================
def run_lgbm(X, y, X_test, cat_cols, seed: int, params: dict, tag: str):
    if already_done(tag):
        log(f"[LGBM] SKIP {tag} (cached)")
        return

    X_l = X.copy()
    X_test_l = X_test.copy()
    for c in cat_cols:
        if c in X_l.columns:
            X_l[c] = X_l[c].astype("category")
            X_test_l[c] = X_test_l[c].astype("category")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_l), dtype=float)
    test_pred = np.zeros(len(X_test_l), dtype=float)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_l), 1):
        X_tr, X_va = X_l.iloc[tr_idx], X_l.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, free_raw_data=False)
        valid_set = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, free_raw_data=False)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=20000,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=300, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        va_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof[va_idx] = va_pred
        test_pred += model.predict(X_test_l, num_iteration=model.best_iteration) / N_SPLITS
        log(f"[LGBM] {tag} | Fold {fold} RMSE: {rmse(y_va, va_pred):.6f} | best_iter={model.best_iteration}")

    cv = rmse(y, oof)
    meta = {"family": "lgbm", "seed": seed, "cv_rmse": cv, "params": params}
    save_preds(tag, oof, test_pred, meta)
    log(f"[LGBM] DONE {tag} | CV RMSE: {cv:.6f} | time: {time.time()-t0:.1f}s")
    gc.collect()

# =========================
# XGBOOST RUNS
# =========================
def run_xgb(X, y, X_test, cat_cols, seed: int, params: dict, tag: str):
    if already_done(tag):
        log(f"[XGB] SKIP {tag} (cached)")
        return

    # one-hot
    X_all = pd.concat([X, X_test], axis=0).copy()
    for c in cat_cols:
        if c in X_all.columns:
            X_all[c] = X_all[c].astype("string").fillna("Missing")
    X_all = pd.get_dummies(X_all, columns=[c for c in cat_cols if c in X_all.columns], dummy_na=False)

    X_xgb = X_all.iloc[: len(X), :]
    X_test_xgb = X_all.iloc[len(X):, :]

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_xgb), dtype=float)
    test_pred = np.zeros(len(X_test_xgb), dtype=float)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_xgb), 1):
        X_tr, X_va = X_xgb.iloc[tr_idx], X_xgb.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        dte = xgb.DMatrix(X_test_xgb)

        booster = xgb.train(
            params,
            dtr,
            num_boost_round=50000,
            evals=[(dva, "valid")],
            verbose_eval=False,
            early_stopping_rounds=500,
        )

        va_pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        oof[va_idx] = va_pred
        test_pred += booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)) / N_SPLITS
        log(f"[XGB] {tag} | Fold {fold} RMSE: {rmse(y_va, va_pred):.6f} | best_iter={booster.best_iteration}")

    cv = rmse(y, oof)
    meta = {"family": "xgb", "seed": seed, "cv_rmse": cv, "params": params}
    save_preds(tag, oof, test_pred, meta)
    log(f"[XGB] DONE {tag} | CV RMSE: {cv:.6f} | time: {time.time()-t0:.1f}s")
    gc.collect()

# =========================
# TABULAR NN (Embedding MLP)
# =========================
class TabDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = X_cat
        self.X_num = X_num
        self.y = y

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_cat[idx], self.X_num[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

class EmbMLP(nn.Module):
    def __init__(self, cat_cardinalities, num_dim, emb_dim_rule="auto", hidden=[512, 256, 128], dropout=0.15):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities

        # embedding dims
        emb_dims = []
        for card in cat_cardinalities:
            if emb_dim_rule == "auto":
                d = int(min(64, round(1.6 * (card ** 0.56))))
                d = max(4, d)
            else:
                d = int(emb_dim_rule)
            emb_dims.append(d)

        self.emb_layers = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, emb_dims)
        ])

        emb_total = sum(emb_dims)
        in_dim = emb_total + num_dim

        layers = []
        cur = in_dim
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            cur = h
        layers += [nn.Linear(cur, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(embs + [x_num], dim=1)
        out = self.mlp(x).squeeze(1)
        return out

def encode_for_nn(X: pd.DataFrame, X_test: pd.DataFrame, cat_cols, num_cols):
    # create category -> code mapping using combined (train+test) for stability
    X_all = pd.concat([X[cat_cols], X_test[cat_cols]], axis=0).copy()
    cat_maps = {}
    cat_cardinalities = []
    for c in cat_cols:
        vals = X_all[c].astype("string").fillna("Missing").unique().tolist()
        # stable mapping
        cat_maps[c] = {v: i for i, v in enumerate(vals)}
        cat_cardinalities.append(len(vals))

    def map_cat(df):
        arr = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            m = cat_maps[c]
            arr[:, j] = df[c].astype("string").fillna("Missing").map(m).astype(np.int64).values
        return arr

    X_cat = map_cat(X)
    X_test_cat = map_cat(X_test)

    # numeric as float32; simple standardization (helps NN)
    X_num = X[num_cols].astype(np.float32).values
    X_test_num = X_test[num_cols].astype(np.float32).values
    mean = X_num.mean(axis=0, keepdims=True)
    std = X_num.std(axis=0, keepdims=True) + 1e-6
    X_num = (X_num - mean) / std
    X_test_num = (X_test_num - mean) / std

    return X_cat, X_num, X_test_cat, X_test_num, cat_cardinalities

def run_nn(X, y, X_test, cat_cols, num_cols, seed: int, tag: str):
    if already_done(tag):
        log(f"[NN] SKIP {tag} (cached)")
        return

    seed_everything(seed)

    X_cat, X_num, X_test_cat, X_test_num, cat_cardinalities = encode_for_nn(X, X_test, cat_cols, num_cols)
    y_arr = y.astype(np.float32).values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)

    # NN hyperparams (reasonable “all out” defaults)
    batch_size = 4096 if DEVICE == "cuda" else 1024
    lr = 2e-3
    max_epochs = 25
    patience = 4
    weight_decay = 1e-4

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cat), 1):
        log(f"\n[NN] {tag} | Fold {fold} | device={DEVICE}")

        tr_ds = TabDataset(X_cat[tr_idx], X_num[tr_idx], y_arr[tr_idx])
        va_ds = TabDataset(X_cat[va_idx], X_num[va_idx], y_arr[va_idx])
        te_ds = TabDataset(X_test_cat, X_test_num, None)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(DEVICE=="cuda"))
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

        model = EmbMLP(cat_cardinalities=cat_cardinalities, num_dim=len(num_cols)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_rmse = 1e9
        best_state = None
        bad = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for xb_cat, xb_num, yb in tr_loader:
                xb_cat = xb_cat.to(DEVICE, non_blocking=True)
                xb_num = xb_num.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    pred = model(xb_cat, xb_num)
                    loss = loss_fn(pred, yb)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            # validate
            model.eval()
            preds = []
            ys = []
            with torch.no_grad():
                for xb_cat, xb_num, yb in va_loader:
                    xb_cat = xb_cat.to(DEVICE, non_blocking=True)
                    xb_num = xb_num.to(DEVICE, non_blocking=True)
                    pred = model(xb_cat, xb_num).detach().cpu().numpy()
                    preds.append(pred)
                    ys.append(yb.numpy())
            preds = np.concatenate(preds)
            ys = np.concatenate(ys)
            val_rmse = rmse(ys, preds)

            log(f"[NN] {tag} | Fold {fold} | Epoch {epoch:02d} | val RMSE {val_rmse:.6f}")

            if val_rmse < best_rmse - 1e-4:
                best_rmse = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        # load best, predict val + test
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            # val
            vpreds = []
            for xb_cat, xb_num, yb in va_loader:
                xb_cat = xb_cat.to(DEVICE, non_blocking=True)
                xb_num = xb_num.to(DEVICE, non_blocking=True)
                vpreds.append(model(xb_cat, xb_num).detach().cpu().numpy())
            vpreds = np.concatenate(vpreds)
            oof[va_idx] = vpreds.astype(np.float32)

            # test
            tpreds = []
            for xb_cat, xb_num in te_loader:
                xb_cat = xb_cat.to(DEVICE, non_blocking=True)
                xb_num = xb_num.to(DEVICE, non_blocking=True)
                tpreds.append(model(xb_cat, xb_num).detach().cpu().numpy())
            tpreds = np.concatenate(tpreds).astype(np.float32)
            test_pred += tpreds / N_SPLITS

        log(f"[NN] {tag} | Fold {fold} best RMSE: {best_rmse:.6f}")
        del model, opt
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    cv = rmse(y_arr, oof)
    meta = {"family": "nn_embmlp", "seed": seed, "cv_rmse": cv, "device": DEVICE}
    save_preds(tag, oof.astype(np.float64), test_pred.astype(np.float64), meta)
    log(f"[NN] DONE {tag} | CV RMSE: {cv:.6f} | time: {time.time()-t0:.1f}s")

# =========================
# META STACKING (OOF-safe)
# =========================
def meta_cv_predict(X_oof, y, X_test, model, seed=42):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_meta = np.zeros(len(y), dtype=float)
    test_meta = np.zeros(X_test.shape[0], dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_oof), 1):
        X_tr, X_va = X_oof[tr_idx], X_oof[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        mdl = model
        mdl.fit(X_tr, y_tr)
        oof_meta[va_idx] = mdl.predict(X_va)
        test_meta += mdl.predict(X_test) / N_SPLITS

    return oof_meta, test_meta

def write_submission(test_ids, pred, filename):
    df = pd.DataFrame({ID_COL: test_ids, TARGET: pred})
    path = os.path.join(SUB_DIR, filename)
    df.to_csv(path, index=False)
    log(f"Saved {path} ✅")

# =========================
# MAIN
# =========================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X, y, X_test, cat_cols, num_cols = prepare_features(train_df, test_df)
test_ids = test_df[ID_COL].values
y_np = y.values.astype(float)

log(f"Device: {DEVICE} (AMP={USE_AMP})")
log(f"Train: {X.shape}  Test: {X_test.shape}  Cat cols: {len(cat_cols)}")

# --- LGBM param sets (3 families)
LGBM_A = dict(
    objective="regression", metric="rmse",
    learning_rate=0.03,
    num_leaves=256,
    max_depth=-1,
    min_data_in_leaf=50,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    bagging_freq=1,
    lambda_l2=5.0,
    verbose=-1,
)
LGBM_B = dict(
    objective="regression", metric="rmse",
    learning_rate=0.03,
    num_leaves=512,
    max_depth=-1,
    min_data_in_leaf=80,
    feature_fraction=0.75,
    bagging_fraction=0.75,
    bagging_freq=1,
    lambda_l2=8.0,
    verbose=-1,
)
# more aggressive family (often helps synthetic)
LGBM_C = dict(
    objective="regression", metric="rmse",
    learning_rate=0.03,
    num_leaves=1024,
    max_depth=-1,
    min_data_in_leaf=120,
    feature_fraction=0.70,
    bagging_fraction=0.70,
    bagging_freq=1,
    lambda_l2=12.0,
    verbose=-1,
)

# --- XGB params (CPU stable; you can switch to gpu_hist later if you want)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    eta=0.03,
    max_depth=10,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=8.0,
    reg_alpha=0.0,
    tree_method="gpu_hist" if DEVICE == "cuda" else "hist",
    seed=0,
)

# 1) Train model library
# LGBM families
for s in SEEDS_LGBM:
    run_lgbm(X, y, X_test, cat_cols, s, LGBM_A, f"lgbm_A_seed{s}")
for s in SEEDS_LGBM:
    run_lgbm(X, y, X_test, cat_cols, s, LGBM_B, f"lgbm_B_seed{s}")
for s in SEEDS_LGBM:
    run_lgbm(X, y, X_test, cat_cols, s, LGBM_C, f"lgbm_C_seed{s}")

# XGB family
for s in SEEDS_XGB:
    p = dict(XGB_PARAMS)
    p["seed"] = s
    run_xgb(X, y, X_test, cat_cols, s, p, f"xgb_seed{s}")

# NN family (big diversity)
for s in SEEDS_NN:
    run_nn(X, y, X_test, cat_cols, num_cols, s, f"nn_embmlp_seed{s}")

# 2) Collect all cached runs
tags = list_tags()
metas = [load_meta(t) for t in tags]
cvs = np.array([m["cv_rmse"] for m in metas])
order = np.argsort(cvs)

log("\nTop 15 runs by CV:")
for i in range(min(15, len(order))):
    idx = order[i]
    log(f"{i+1:2d}. {tags[idx]} | CV {cvs[idx]:.6f} | {metas[idx]['family']} seed={metas[idx].get('seed')}")

# 3) Build OOF/test matrix
oofs = []
tests = []
for t in tags:
    oof, te = load_preds(t)
    oofs.append(oof)
    tests.append(te)
X_oof = np.vstack(oofs).T
X_te  = np.vstack(tests).T

# 4) Meta models with OOF-safe meta-CV
meta_oofs = {}
meta_tests = {}

for name, model in META_MODELS.items():
    log(f"\n[STACK] Training meta model: {name}")
    oof_m, te_m = meta_cv_predict(X_oof, y_np, X_te, model, seed=42)
    meta_oofs[name] = oof_m
    meta_tests[name] = te_m
    log(f"[STACK] {name} meta OOF RMSE: {rmse(y_np, oof_m):.6f}")

# 5) Second-level blending of meta models using OOF search
meta_names = list(meta_oofs.keys())
M_oof = np.vstack([meta_oofs[n] for n in meta_names]).T
M_te  = np.vstack([meta_tests[n] for n in meta_names]).T

best = (1e9, None)
# small grid over weights for 3 models (ridge/enet/bayes)
grid = np.linspace(0, 1, 21)
for w0 in grid:
    for w1 in grid:
        w2 = 1.0 - w0 - w1
        if w2 < 0:
            continue
        w = np.array([w0, w1, w2])
        pred = M_oof @ w
        score = rmse(y_np, pred)
        if score < best[0]:
            best = (score, w)

best_rmse, best_w = best
log(f"\n[STACK2] Best meta-blend OOF RMSE: {best_rmse:.6f} weights={dict(zip(meta_names, best_w.round(3)))}")

final_stack = (M_te @ best_w)

# 6) Also output simple strong baselines
mean_all = X_te.mean(axis=1)
topk = min(20, X_te.shape[1])
X_te_topk = X_te[:, order[:topk]]
mean_topk = X_te_topk.mean(axis=1)

# 7) Write submissions
write_submission(test_ids, final_stack, "submission_stack2.csv")      # best guess
write_submission(test_ids, meta_tests["ridge"], "submission_ridge.csv")
write_submission(test_ids, meta_tests["bayes"], "submission_bayes.csv")
write_submission(test_ids, mean_all, "submission_mean_all.csv")
write_submission(test_ids, mean_topk, f"submission_mean_top{topk}.csv")

log("\nDONE. Upload subs/*.csv to Kaggle and see what wins.")
