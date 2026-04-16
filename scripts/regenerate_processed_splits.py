from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RANDOM_STATE = 42


def standardize_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def load_raw_dataset() -> pd.DataFrame:
    frames = []
    for file_path in sorted(RAW_DATA_DIR.glob("*.csv")):
        df = standardize_csv(file_path)
        frames.append(df)
        print(f"Loaded {file_path.name} with shape {df.shape}")
    if not frames:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DATA_DIR}")
    return pd.concat(frames, ignore_index=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip().apply(
        lambda x: 0 if x == "BENIGN" else 1
    )
    return df


def save_split(X: pd.DataFrame, y: pd.Series, split: str) -> None:
    split_dir = PROCESSED_DATA_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(split_dir / f"X_{split}.csv", index=False)
    y.to_csv(split_dir / f"y_{split}.csv", index=False, header=["label"])


def main() -> None:
    df = clean_dataset(load_raw_dataset())

    X = df.drop(columns="label")
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    save_split(X_train, y_train, "train")
    save_split(X_val, y_val, "val")
    save_split(X_test, y_test, "test")

    for split, labels in {
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }.items():
        print(split, labels.value_counts().to_dict())


if __name__ == "__main__":
    main()
