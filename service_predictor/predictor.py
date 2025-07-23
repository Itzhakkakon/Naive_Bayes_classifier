import pandas as pd
class ModelPredictor:
    def __init__(self, model_weights):
        self._model_weights = model_weights

    def predict(self, row: pd.Series) -> str:
        scores = {}
        for cls, cls_data in self._model_weights.items():
            score = cls_data['__prior__']
            for feature, value in row.items():
                score += cls_data.get(feature, {}).get(value, 0)
            scores[cls] = score
        return max(scores, key=scores.get)

