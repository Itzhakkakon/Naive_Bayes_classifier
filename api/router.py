from fastapi import APIRouter, Request, HTTPException
import pandas as pd

# ניצור "ראוטר" - זהו אובייקט שמקבץ את כל נקודות הקצה שקשורות אליו
router = APIRouter()


# הפונקציה הזו תוגדר כך שהיא תקבל את התלות שלה (predictor וכו')
# מהקובץ הראשי שיקרא לה
def setup_dependencies(app_predictor, app_feature_names):
    @router.get("/classify")
    def classify_endpoint(request: Request):
        instance_data = dict(request.query_params)

        missing_features = [
            feature for feature in app_feature_names if feature not in instance_data
        ]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={"error": "Missing features", "missing": missing_features},
            )
        try:
            instance_as_series = pd.Series(instance_data)
            prediction = app_predictor.predict(instance_as_series)
            return {"input_features": instance_data, "prediction": prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))