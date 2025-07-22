import pandas as pd

def clean_data(df):
    df = df.loc[:, df.nunique() > 1]
    df = df.drop(columns=['stalk-root'], errors='ignore')
    return df