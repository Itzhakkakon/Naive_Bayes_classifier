from service_data_loader.data_loader import load_data
from pprint import pprint
from service_model.naive_bayes_model import NaiveBayesModel
from service_preprocessor.data_splitter import split_data

if __name__ == "__main__":
    data = load_data()
    target = data.columns.tolist()[-1]
    train_data, test_data = split_data(data, target)
    model = NaiveBayesModel(alpha = 1.0)
    model_weights = model.build_model(train_data, target)

    print("מודל Naive Bayes נבנה בהצלחה")
    pprint(model)
