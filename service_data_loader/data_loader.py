import pandas as pd
def load_data(path):
    df: pd.DataFrame = pd.read_csv(path)
    return df