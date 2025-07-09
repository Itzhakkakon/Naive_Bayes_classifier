import pandas as pd
from copy import deepcopy

def load_data():
    df = pd.read_csv('mushroom_decoded.csv')
    df = df.loc[:, df.nunique() > 1]
    df = df.drop(columns=['stalk-root'], errors='ignore')
    return df


def target_column_division(df):
    column = df.drop(columns=['poisonous'])
    target = df['poisonous']
    return column, target


def dictionary_temporary(df):
    temp_dict = {}
    for col in df.columns:
        temp_dict[col] = {}
        for v in col.unique():
            temp_dict[col][v] = 0
    return temp_dict

def dictionary_weights(df):
    weights = {}
    for col in target.unique():
        weights[col] = deepcopy(dictionary_temporary(df[col]))




if __name__ == "__main__":
    data = load_data()
    column, target = target_column_division(data)














# def get_columns(df):
#     columns = df.columns.tolist()
#     columns.remove('poisonous')
#     return columns




# def Target_column_division(df):
    # X = df.drop(columns=['poisonous'])
    # y = df['poisonous']
    # return X, y

# def convert_to_weights(df):
#     weights = {}
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             unique_values = df[col].unique()
#             weights[col] = {value: i + 1 for i, value in enumerate(unique_values)}
#     return weights
#


