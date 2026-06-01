import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "students.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model",   "student_model.pkl")

FEATURE_COLS = [
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "PracticeTests",
    "SleepHours",
    "SocialMediaHours",
]
TARGET_COL = "FinalMarks"


def train_model():
    """
    Load students.csv  →  train LinearRegression  →  save .pkl
    Returns (model, accuracy_dict) or raises an exception on failure.
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please make sure 'students.csv' is inside the 'dataset/' folder."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset is empty. Please add records to students.csv.")

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

        df = df.dropna()

    if len(df) < 10:
        raise ValueError("Dataset has too few clean records (need at least 10).")

    print(f"\n  Dataset loaded successfully  →  {len(df)} records found.")

    X = df[FEATURE_COLS].values    
    y = df[TARGET_COL].values     

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)         

    y_pred    = model.predict(X_test)
    r2        = r2_score(y_test, y_pred)        
    mae       = mean_absolute_error(y_test, y_pred)
    accuracy  = round(r2 * 100, 2)               

    accuracy_info = {
        "r2_score"  : round(r2,  4),
        "accuracy"  : accuracy,          
        "mae"       : round(mae, 2),     
        "train_size": len(X_train),
        "test_size" : len(X_test),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"  Model trained and saved  →  model/student_model.pkl")
    return model, accuracy_info