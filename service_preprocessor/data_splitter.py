from sklearn.model_selection import train_test_split

def split_data(data, target_column, test_size=0.3):
    train_df, test_df = train_test_split(data, test_size=test_size, stratify=data[target_column], random_state=42)
    return train_df, test_df
