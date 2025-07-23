import pandas as pd

def read_csv(file_path):
    return pd.read_csv(file_path)

#def load_data(csv):
#     df = pd.read_csv(csv)
def load_data():
    df = pd.read_csv('Data/mushroom_decoded.csv')
    return df