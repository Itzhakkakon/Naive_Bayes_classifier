# app.py

from fastapi import FastAPI, Request, HTTPException
import pandas as pd
from manager import build_and_evaluate_model # <-- מייבא את קוד הבנייה
from service_predictor.predictor import ModelPredictor

# --- שלב 1: ביצוע תהליך הבנייה ---
# הקריאה הזו מפעילה את כל הקוד שנמצא ב-manager.py
model, model_weights, target_col_name, feature_names = build_and_evaluate_model()

# --- שלב 2: הכנת השרת ---
predictor = ModelPredictor(model_weights)
app = FastAPI(title="Mushroom Classifier API")

# --- שלב 3: הגדרת נקודת הקצה לחיזוי ---
@app.get("/classify")
def classify_endpoint(request: Request):
    # (הקוד של ה-endpoint מפה והלאה זהה לקוד מהתשובה הקודמת)
    instance_data = dict(request.query_params)
    missing_features = [
        feature for feature in feature_names if feature not in instance_data
    ]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing features", "missing": missing_features},
        )
    try:
        instance_as_series = pd.Series(instance_data)
        prediction = predictor.predict(instance_as_series)
        return {"input_features": instance_data, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- שלב 4: הרצת השרת ---
# ניתן להוסיף את זה אם רוצים להריץ ישירות עם 'python app.py'
if __name__ == "__main__":
    import uvicorn
    print("\n>>> מפעיל שרת וממתין לבקשות סיווג בכתובת http://127.0.0.1:8000 <<<")
    uvicorn.run(app, host="127.0.0.1", port=8000)