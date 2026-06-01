import os
import pickle
import numpy as np
from datetime import datetime

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model",   "student_model.pkl")
HISTORY_PATH  = os.path.join(BASE_DIR, "history", "predictions.txt")


def load_model():
    """Load the saved .pkl model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found!\n"
            "Please run Option 1 (Train Model) first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def get_performance_category(marks):
    """
    Return a label based on predicted marks.
    This helps students understand where they stand.
    """
    if marks >= 85:
        return "EXCELLENT  ★★★★★"
    elif marks >= 70:
        return "GOOD       ★★★★"
    elif marks >= 55:
        return "AVERAGE    ★★★"
    elif marks >= 40:
        return "BELOW AVERAGE ★★"
    else:
        return "NEEDS IMPROVEMENT ★"


def predict_marks(study_hours, attendance, previous_marks,
                  practice_tests, sleep_hours, social_media_hours):
    """
    Accept student data  →  load model  →  return predicted marks.
    Input values are already validated before this function is called.
    """
    model = load_model()

    input_data = np.array([[
        study_hours,
        attendance,
        previous_marks,
        practice_tests,
        sleep_hours,
        social_media_hours,
    ]])

    Predicted = model.predict(input_data)[0]

    predicted = round(float(np.clip(predicted, 0, 100)), 2)

    return predicted


def save_prediction_to_history(inputs, predicted_marks):
    """
    Append one prediction record to history/predictions.txt
    so students can review past predictions later.
    """
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    category  = get_performance_category(predicted_marks)

    line = (
        f"[{timestamp}]  "
        f"Study={inputs[0]}h  "
        f"Attend={inputs[1]}%  "
        f"PrevMarks={inputs[2]}  "
        f"Tests={inputs[3]}  "
        f"Sleep={inputs[4]}h  "
        f"Social={inputs[5]}h  "
        f"→  Predicted={predicted_marks}  "
        f"| {category}\n"
    )

    with open(HISTORY_PATH, "a") as f:
        f.write(line)


def view_history():
    """Print all saved predictions from the history file."""
    if not os.path.exists(HISTORY_PATH):
        print("\n  No prediction history found yet.")
        print("  Make at least one prediction first (Option 2).")
        return

    with open(HISTORY_PATH, "r") as f:
        lines = f.readlines()

    if not lines:
        print("\n  History file is empty.")
        return

    print(f"\n  {'─'*70}")
    print(f"  PREDICTION HISTORY  ({len(lines)} record(s))")
    print(f"  {'─'*70}")
    for i, line in enumerate(lines, 1):
        print(f"  {i:>3}.  {line}", end="")
    print(f"  {'─'*70}")
