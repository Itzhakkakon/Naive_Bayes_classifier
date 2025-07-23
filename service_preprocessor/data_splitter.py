#חותך 30-70
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame, target_column: str, test_size=0.3) -> tuple:
    train_df, test_df = train_test_split(data, test_size=test_size, stratify=data[target_column], random_state=42)
    return train_df, test_df
