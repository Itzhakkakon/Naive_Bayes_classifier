# # app.py
#
# from flask import Flask, request, jsonify
# from manager import build_and_evaluate_model
#
# # =================================================================
# #  חלק 1: טעינת המודל (רץ פעם אחת כשהשרת עולה)
# # =================================================================
# # כאן אנו קוראים למנהל שלנו כדי שיבנה עבורנו את המודל.
# # התוצרים נשמרים במשתנים גלובליים שיהיו זמינים לכל אורך חיי השרת.
# model, model_weights, target_col_name, feature_names = build_and_evaluate_model()
#
#
# # =================================================================
# #  חלק 2: הגדרת והרצת השרת
# # =================================================================
#
# app = Flask(__name__)
#
# @app.route('/classify', methods=['GET'])
# def classify_endpoint():
#     """
#     נקודת קצה שמקבלת מאפייני פטריה כפרמטרים ב-URL ומסווגת אותה.
#     """
#     instance_data = request.args.to_dict()
#
#     if not instance_data:
#         return jsonify({"error": "No data provided. Please provide features as URL parameters."}), 400
#
#     # אימות שכל המאפיינים הדרושים קיימים
#     missing_features = [feature for feature in feature_names if feature not in instance_data]
#     if missing_features:
#         return jsonify({
#             "error": "Missing required features in URL parameters.",
#             "missing": missing_features
#         }), 400
#
#     try:
#         prediction = model.classify(instance_data, model_weights)
#         return jsonify({
#             "input_data": instance_data,
#             "prediction": prediction
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     print("\n>>> מפעיל שרת וממתין לבקשות סיווג בכתובת http://127.0.0.1:5000 <<<")
#     app.run(host='0.0.0.0', port=5000, debug=True)