from data_loader import load_data
from pprint import pprint
from naive_bayes_model import NaiveBayesModel

if __name__ == "__main__":
    data = load_data()
    target = data.columns.tolist()[-1]
    model = NaiveBayesModel(alpha = 1.0)

    print("מודל Naive Bayes נבנה בהצלחה")
    pprint(model)
