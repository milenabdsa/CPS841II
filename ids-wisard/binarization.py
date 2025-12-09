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


def thermometer_encode(X: np.ndarray, n_bits: int = 8) -> np.ndarray:
    n_samples, n_features = X.shape

    thresholds = np.linspace(0, 1, n_bits + 1, endpoint=True)[1:] 

    X_expanded = X[:, :, None]  
    bits = (X_expanded >= thresholds).astype(np.uint8)  


    return bits.reshape(n_samples, n_features * n_bits)
