# from service_data_loader.data_loader import load_data
# from pprint import pprint
# from service_model.naive_bayes_model import NaiveBayesModel
# from service_preprocessor.data_splitter import split_data
#
# if __name__ == "__main__":
#     data = load_data()
#     target = data.columns.tolist()[-1]
#     train_data, test_data = split_data(data, target)
#     model = NaiveBayesModel(alpha = 1.0)
#     model_weights = model.build_model(train_data, target)
#
#     print("מודל Naive Bayes נבנה בהצלחה")
#     pprint(model)
# main.py או server.py

# שלב 0: ייבוא כל הרכיבים הדרושים
from service_data_loader.data_loader import load_data
from service_preprocessor.cleaner import clean_data # <-- לאחר שתעביר אותו לתיקייה הנכונה
from service_preprocessor.data_splitter import split_data
from service_model.naive_bayes_model import NaiveBayesModel
from service_evaluator.evaluator import evaluate
# from flask import Flask, request, jsonify  # <-- תצטרך להתקין ולהוסיף את זה בשלב השרת

# =================================================================
#  חלק 1: בנייה ובדיקה של המודל (רץ פעם אחת בהתחלה)
# =================================================================

print("1. טוען נתונים...")
data = load_data()

print("2. מטפל בנתונים (ניקוי)...")
# שים לב: חשוב להעביר את cleaner.py ל-service_preprocessor
cleaned_data = clean_data(data)

# הגדרת משתנה המטרה
target = cleaned_data.columns.tolist()[-1]

print("3. מפצל נתונים לאימון ובדיקה...")
train_data, test_data = split_data(cleaned_data, target, test_size=0.3)

print("4. בונה ומאמן את מודל Naive Bayes...")
# יצירת מופע של המודל
model = NaiveBayesModel(alpha=1.0)
# אימון המודל על נתוני האימון
model_weights = model.build_model(train_data, target)
print("   המודל נבנה בהצלחה!")

print("5. בודק את ביצועי המודל...")
accuracy = evaluate(model, model_weights, test_data, target)
print(f"   רמת הדיוק של המודל על סט הבדיקה: {accuracy:.2%}")

# # =================================================================
# #  חלק 2: הפעלת השרת והאזנה לבקשות סיווג
# # =================================================================
#
# # בשלב זה, המשתנים 'model' ו-'model_weights' מוכנים לשימוש
# print("\n6. מפעיל שרת וממתין לבקשות סיווג...")
#
# # כאן יבוא הקוד של השרת (לדוגמה, עם Flask)
# # app = Flask(__name__)
#
# # @app.route('/classify', methods=['GET'])
# # def classify_endpoint():
# #     # כאן תקבל נתונים חדשים מהבקשה
# #     # תשתמש ב- model.classify(new_data, model_weights)
# #     # ותחזיר את התוצאה
# #     return jsonify({"prediction": "some_result"})
#
# # if __name__ == '__main__':
# #     # app.run(debug=True)
# #     print("השרת סיים את פעולתו.") # שורה זו תודפס רק אחרי כיבוי השרת