import os
from train_model import train_model
from predict    import (predict_marks, get_performance_category,
                        save_prediction_to_history, view_history)

_accuracy_info = None


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print("\n" + "=" * 44)
    print("    STUDENT MARKS PREDICTION SYSTEM")
    print("=" * 44)


def print_menu():
    print("\n  1. Train Model")
    print("  2. Predict Marks")
    print("  3. View Prediction History")
    print("  4. Show Model Accuracy")
    print("  5. Exit")
    print("\n" + "-" * 44)

def get_float_input(prompt, min_val, max_val):
    """
    Keep asking until the user enters a valid number
    inside [min_val, max_val].
    """
    while True:
        try:
            value = float(input(f"  {prompt} ({min_val}–{max_val}): ").strip())
            if min_val <= value <= max_val:
                return value
            else:
                print(f"  ⚠  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  ⚠  Invalid input. Please enter a number.")


def option_train():
    global _accuracy_info
    print("\n" + "─" * 44)
    print("  TRAINING MODEL …")
    print("─" * 44)
    try:
        model, acc = train_model()
        _accuracy_info = acc
        print("\n  ✔  Training complete!")
        print(f"  ✔  Accuracy  : {acc['accuracy']} %")
        print(f"  ✔  MAE       : {acc['mae']} marks")
        print(f"  ✔  Train set : {acc['train_size']} records")
        print(f"  ✔  Test  set : {acc['test_size']}  records")
    except FileNotFoundError as e:
        print(f"\n  ✘  ERROR: {e}")
    except ValueError as e:
        print(f"\n  ✘  DATA ERROR: {e}")
    except Exception as e:
        print(f"\n  ✘  Unexpected error during training: {e}")


def option_predict():
    print("\n" + "─" * 44)
    print("  ENTER STUDENT DETAILS")
    print("─" * 44)
    print("  (Type a number and press Enter for each field)\n")

    try:
        study_hours        = get_float_input("Study Hours per day      ", 0, 16)
        attendance         = get_float_input("Attendance percentage    ", 0, 100)
        previous_marks     = get_float_input("Previous Exam Marks      ", 0, 100)
        practice_tests     = get_float_input("Practice Tests completed ", 0, 100)
        sleep_hours        = get_float_input("Sleep Hours per day      ", 0, 12)
        social_media_hours = get_float_input("Social Media Hours/day   ", 0, 12)

        
        predicted = predict_marks(
            study_hours, attendance, previous_marks,
            practice_tests, sleep_hours, social_media_hours
        )
        category = get_performance_category(predicted)

        
        print("\n" + "=" * 44)
        print("  PREDICTION RESULT")
        print("=" * 44)
        print(f"\n  Predicted Final Marks : {predicted} / 100")
        print(f"  Performance Category  : {category}")
        print("\n" + "=" * 44)

        
        inputs = [study_hours, attendance, previous_marks,
                  practice_tests, sleep_hours, social_media_hours]
        save_prediction_to_history(inputs, predicted)
        print("  ✔  Result saved to history/predictions.txt")

    except FileNotFoundError as e:
        print(f"\n  ✘  {e}")
    except Exception as e:
        print(f"\n  ✘  Prediction failed: {e}")


def option_history():
    try:
        view_history()
    except Exception as e:
        print(f"\n  ✘  Could not read history: {e}")


def option_accuracy():
    global _accuracy_info
    print("\n" + "─" * 44)
    print("  MODEL ACCURACY REPORT")
    print("─" * 44)

    if _accuracy_info is None:
        print("\n  No accuracy data available yet.")
        print("  Please train the model first (Option 1).")
        return

    acc = _accuracy_info
    print(f"\n  R² Score   : {acc['r2_score']}  (1.0 = perfect)")
    print(f"  Accuracy   : {acc['accuracy']} %")
    print(f"  MAE        : {acc['mae']} marks  (avg prediction error)")
    print(f"  Train set  : {acc['train_size']} records")
    print(f"  Test  set  : {acc['test_size']}  records")
    print()
    print("  ── What these numbers mean ──────────────────")
    print("  R² Score tells how well the model fits data.")
    print("  MAE tells how far off predictions are on avg.")
    print("  Lower MAE = better predictions.")
    print("─" * 44)



def main():
    while True:
        clear()
        print_banner()
        print_menu()

        choice = input("  Enter your choice (1–5): ").strip()

        if choice == "1":
            option_train()
        elif choice == "2":
            option_predict()
        elif choice == "3":
            option_history()
        elif choice == "4":
            option_accuracy()
        elif choice == "5":
            print("\n  Thank you for using Student Marks Prediction System!")
            print("  Goodbye!\n")
            break
        else:
            print("\n  ⚠  Invalid choice. Please enter a number between 1 and 5.")

        input("\n  Press Enter to return to the menu …")

if __name__ == "__main__":
    main()