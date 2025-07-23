#המסווג
import pandas as pd

class ModelPredictor:
    # האתחול מקבל את המשקולות ושומר אותן
    def __init__(self, model_weights: dict):
        self._model_weights = model_weights

    # המתודה predict מקבלת רק את השורה לסיווג
    def predict(self, row: pd.Series) -> str:
        scores = {}
        for cls, cls_data in self._model_weights.items():
            score = cls_data['__prior__']
            for feature, value in row.items():
                # השורה הזו טובה וחסינה לשגיאות אם יש מאפיין חסר במודל
                score += cls_data.get(feature, {}).get(value, 0)
            scores[cls] = score
        return max(scores, key=scores.get)