import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/dataset.csv"
TARGET_COLUMN = "Label"

def clip_outliers(df: pd.DataFrame, quantile: float = 0.999) -> pd.DataFrame:

    df_clipped = df.copy()
    numeric_cols = df_clipped.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        q = df_clipped[col].quantile(quantile)
        df_clipped[col] = df_clipped[col].clip(upper=q)

    return df_clipped

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)

    cols_to_drop = [
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Timestamp",
    ]
    cols_to_drop += [c for c in df.columns if c.lower().startswith("unnamed")]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()


    df = clip_outliers(df, quantile=0.999)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y

def split_dataset(X, y, test_size=0.3):

    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test = le.transform(y_test_str)

    return X_train, X_test, y_train, y_test, le
