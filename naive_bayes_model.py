import pandas as pd
from copy import deepcopy
from math import log

class NaiveBayesModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def build_model(self, train_data: pd.DataFrame, target_col: str):
        df = train_data
        counts_model = self.model_naive_bayes(df, target_col)
        weights_model = self._convert_counts_to_weights(counts_model)
        return weights_model

    def model_naive_bayes(self,data: pd.DataFrame, target: str):
        target_counts = {}
        for val in data[target].unique():
            target_counts[val] = 0

        feature_cols = []
        for col in data.columns:
            if col == target:
                continue
            feature_cols.append(col)
        value_unique = {}
        for v in feature_cols:
            value_unique[v] = {}
            for val in data[v].unique():
                value_unique[v][val] = 0

        likelihoods = {}
        for v in data[target].unique():
            likelihoods[v] = deepcopy(value_unique)

        for _, row in data.iterrows():
            target_value = row[target]
            target_counts[target_value] += 1
            for col in feature_cols:
                likelihoods[target_value][col][row[col]] += 1

        result = {
            "likelihoods": likelihoods,
            "target_counts": target_counts,
            "total_count": len(data)
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
