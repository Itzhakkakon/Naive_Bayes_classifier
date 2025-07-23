from fastapi import FastAPI
import uvicorn
from manager import build_and_evaluate_model
from api.router import router, setup_dependencies

# --- שלב 1: ביצוע תהליך הבנייה ---
# הקוד הזה נשאר זהה
predictor, model_weights, target_col_name, feature_names = build_and_evaluate_model()

# --- שלב 2: הגדרת הראוטר עם התלויות הדרושות ---
# כאן אנו "מזריקים" את ה-predictor ואת שמות המאפיינים
# לתוך קובץ הראוטר שלנו, כדי שנקודת הקצה תוכל להשתמש בהם.
setup_dependencies(app_predictor=predictor, app_feature_names=feature_names)

# --- שלב 3: הרכבת האפליקציה ---
app = FastAPI(title="Mushroom Classifier API")
# כאן אנו "מחברים" את הראוטר שהגדרנו לאפליקציה הראשית
app.include_router(router)

# --- שלב 4: הרצת השרת ---
if __name__ == "__main__":
    print("\n>>> מפעיל שרת וממתין לבקשות סיווג בכתובת http://127.0.0.1:8000 <<<")
    uvicorn.run(app, host="127.0.0.1", port=8000)