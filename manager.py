from service_data_loader.data_loader import load_data
from service_preprocessor.cleaner import clean_data
from service_preprocessor.data_splitter import split_data
from service_model.naive_bayes_model import NaiveBayesModel
from service_evaluator.evaluator import evaluate
from service_predictor.predictor import ModelPredictor



def build_and_evaluate_model():
    """
   טעינה -> ניקוי -> פיצול -> אימון -> הערכה.
    מריץ את כל תהליך בניית המודל, מהתחלה ועד הסוף.
    - טוען נתונים
    - מנקה אותם
    - מפצל לאימון ובדיקה
    - מאמן את המודל
    - בודק את ביצועיו

    :return: מחזיר tuple עם כל הנכסים הדרושים להרצת השרת:
             (אובייקט המודל, משקולות המודל, שם עמודת המטרה, רשימת מאפיינים)
    """
    print("--- מתחיל תהליך בניית המודל ---")

    print("1. טוען נתונים...")
    data_file_path = 'Data/mushroom_decoded.csv'
    data = load_data(data_file_path)

    print("2. מטפל בנתונים (ניקוי)...")
    cleaned_data = clean_data(data)

    target_col = cleaned_data.columns.tolist()[-1]
    feature_cols = cleaned_data.drop(columns=[target_col]).columns.tolist()

    print("3. מפצל נתונים לאימון ובדיקה...")
    train_data, test_data = split_data(cleaned_data, target_col, test_size=0.3)

    print("4. בונה ומאמן את מודל Naive Bayes...")
    model = NaiveBayesModel(alpha=1.0)
    model_weights = model.build_model(train_data, target_col)
    print("   המודל נבנה בהצלחה!")

    predictor = ModelPredictor(model_weights)

    print("5. בודק את ביצועי המודל...")
    accuracy = evaluate(predictor, test_data, target_col)
    print(f"   רמת הדיוק של המודל על סט הבדיקה: {accuracy:.2%}")

    print("--- תהליך בניית המודל הסתיים ---")

    return predictor, model_weights, target_col, feature_cols