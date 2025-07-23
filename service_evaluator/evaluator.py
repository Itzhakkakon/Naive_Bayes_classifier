import pandas as pd


def evaluate(model_instance, model_weights, test_data: pd.DataFrame, target_column: str):


    # הפרדת המאפיינים (features) מהמטרה (target) בסט הבדיקה
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    correct_predictions = 0

    # לולאה שעוברת על כל דגימה בסט הבדיקה
    for i in range(len(X_test)):
        # קבלת שורת נתונים אחת (מאפיינים) כדי לסווג
        row_to_classify = X_test.iloc[i].to_dict()

        # קבלת התווית האמיתית של אותה שורה
        true_label = y_test.iloc[i]

        # שימוש במודל כדי לקבל חיזוי
        predicted_label = model_instance.classify(row_to_classify, model_weights)

        # בדיקה אם החיזוי היה נכון
        if predicted_label == true_label:
            correct_predictions += 1

    # חישוב הדיוק
    accuracy = correct_predictions / len(test_data)

    return accuracy