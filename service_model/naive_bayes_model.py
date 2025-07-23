import pandas as pd
from copy import deepcopy
from math import log

class NaiveBayesModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._df: pd.DataFrame | None = None
        self._tc: str | None = None

    def build_model(self, train_data: pd.DataFrame, target_col: str):
        self._df = train_data
        self._tc = target_col
        counts_model = self._model_naive_bayes()
        weights_model = self._convert_counts_to_weights(counts_model)
        return weights_model

    def _model_naive_bayes(self):
        target_counts = {}
        for val in self._df[self._tc].unique():
            target_counts[val] = 0

        feature_cols = []
        for col in self._df.columns:
            if col == self._tc:
                continue
            feature_cols.append(col)
        value_unique = {}
        for v in feature_cols:
            value_unique[v] = {}
            for val in self._df[v].unique():
                value_unique[v][val] = 0

        likelihoods = {}
        for v in self._df[self._tc].unique():
            likelihoods[v] = deepcopy(value_unique)

        for _, row in self._df.iterrows():
            target_value = row[self._tc]
            target_counts[target_value] += 1
            for col in feature_cols:
                likelihoods[target_value][col][row[col]] += 1

        result = {
            "likelihoods": likelihoods,
            "target_counts": target_counts,
            "total_count": len(self._df)
        }
        return result

    def _convert_counts_to_weights(self, counts_model: dict):
        likelihoods = counts_model["likelihoods"]
        target_counts = counts_model["target_counts"]
        total_count = counts_model["total_count"]
        weights = {}
        for target_value, column_dict in deepcopy(likelihoods).items():
            weights[target_value] = {}

            prior = log(target_counts[target_value] / total_count)
            weights[target_value]["__prior__"] = prior

            for column, value_dict in column_dict.items():
                if any(v == 0 for v in value_dict.values()):
                    value_dict = {k: v + self.alpha for k, v in value_dict.items()}

                total = sum(value_dict.values())

                weights[target_value][column] = {
                    k: log(v / total) for k, v in value_dict.items()
                }
        return weights

