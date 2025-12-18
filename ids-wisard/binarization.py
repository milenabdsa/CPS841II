import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_features(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    df_proc = X.copy()

    if cat_cols:
        df_cat = pd.get_dummies(df_proc[cat_cols].astype(str), drop_first=False)
        df_num = df_proc[numeric_cols]
        df_proc = pd.concat([df_num, df_cat], axis=1)
    else:
        df_proc = df_proc[numeric_cols]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_proc.values.astype(np.float32))

    return X_scaled, scaler, df_proc.columns.tolist()


def apply_preprocess_to_test(X_test: pd.DataFrame, scaler, feature_order):
    numeric_cols = X_test.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_test.columns if c not in numeric_cols]

    df_proc = X_test.copy()

    if cat_cols:
        df_cat = pd.get_dummies(df_proc[cat_cols].astype(str), drop_first=False)
        df_num = df_proc[numeric_cols]
        df_proc = pd.concat([df_num, df_cat], axis=1)
    else:
        df_proc = df_proc[numeric_cols]

    for col in feature_order:
        if col not in df_proc.columns:
            df_proc[col] = 0

    df_proc = df_proc[feature_order]

    X_scaled = scaler.transform(df_proc.values.astype(np.float32))
    return X_scaled


def onehot_binning_encode(X: np.ndarray, n_bins: int = 8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape

    X = np.clip(X, 0.0, 1.0)

    bin_idx = np.floor(X * n_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    out = np.zeros((n_samples, n_features * n_bins), dtype=np.uint8)

    for j in range(n_features):
        out[np.arange(n_samples), j * n_bins + bin_idx[:, j]] = 1

    return out