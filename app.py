import pandas as pd
from copy import deepcopy

def read_csv(file_path):
    return pd.read_csv(file_path)

#def load_data(csv):
#     df = pd.read_csv(csv)
def load_data():
    df = pd.read_csv('mushroom_decoded.csv')
    df = df.loc[:, df.nunique() > 1]
    df = df.drop(columns=['stalk-root'], errors='ignore')
    return df


def model_naive_bayes(data: pd.DataFrame, target: str):
    my_dict = {}
    for col in data.columns:
        if col == target:
            continue
        my_dict[col] = {}
        for v in data[col].unique():
            my_dict[col][v] = 0
    result = {}
    for v in data[target].unique():
        result[v] = deepcopy(my_dict)

    gr = data.groupby(target)
    for name, group in gr:
        for col in group.columns:
            if col == target:
                continue
            for v in group[col].unique():
                result[name][col][v] = (group[col] == v).sum() / len(group)
    return result



# dict1 = {
#         'no': {'age': {'low': 0.5, 'medium': 0.3, 'high': 0.2}
#             ,'income': {'low': 0.4, 'medium': 0.4, 'high': 0.2}}
#
#         ,'yes':{'age': {'low': 0.4, 'medium': 0.8, 'high': 0.9}
#             ,'income': {'low': 0.1, 'medium': 0.6, 'high': 0.3}}
#         }


if __name__ == "__main__":
    data = load_data()
    target = data.columns.tolist()[-1]

    e_model = model_naive_bayes(data, target)