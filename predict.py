import joblib
X_train, X_test, y_train, y_test = joblib.load(
    "models/data.pkl"
)
# Load models
svm_model = joblib.load("models/svm_model.pkl")
lgb_model = joblib.load("models/lgb_model.pkl")
# Load scaler
scaler = joblib.load("models/scaler.pkl")
# Take one test website
sample = X_test.iloc[[0]]
# SVM needs scaled data
sample_scaled = scaler.transform(sample)
# Make predictions
svm_prediction = svm_model.predict(sample_scaled)[0]
lgb_prediction = lgb_model.predict(sample)[0]
def convert_label(result):
    if result == 1:
        return "PHISHING"
    else:
        return "LEGITIMATE"
print("--------------------------------")
print("PHISHING WEBSITE DETECTOR")
print("--------------------------------")

print("SVM Prediction:",
      convert_label(svm_prediction))

print("LightGBM Prediction:",
      convert_label(lgb_prediction))

print("Actual Result:",
      convert_label(y_test.iloc[0]))

print("--------------------------------")