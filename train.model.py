import joblib
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
#load the process data
print("Loading processed data...")
X_train, X_test, y_train, y_test = joblib.load(
    "models/data.pkl"
)
scaler = joblib.load(
    "models/scaler.pkl"
)
print("Data loaded successfully!")
#Data for SVM
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed.")
print("\nTraining SVM model...")
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced"
)
svm_model.fit(
    X_train_scaled,
    y_train
)
print("SVM training completed!")
#SVM predictions
svm_predictions = svm_model.predict(
    X_test_scaled
)
#Train LightGBM model
print("\nTraining LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)
lgb_model.fit(
    X_train,
    y_train
)
print("LightGBM training completed!")
#LightGBM predictions
lgb_predictions = lgb_model.predict(
    X_test
)
#evaluation function
def evaluate_model(name, actual, predicted):

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            actual,
            predicted
        )
    )
#evaluating the problem
evaluate_model(
    "SVM RESULTS",
    y_test,
    svm_predictions
)
#evaluate the GBM
evaluate_model(
    "LIGHTGBM RESULTS",
    y_test,
    lgb_predictions
)
#save the Model
joblib.dump(
    svm_model,
    "models/svm_model.pkl"
)
joblib.dump(
    lgb_model,
    "models/lgb_model.pkl"
)
#Done 

print("\n")
print("=" * 50)
print("MODEL TRAINING COMPLETED!")
print("=" * 50)

print("\nSaved models:")
print("models/svm_model.pkl")
print("models/lgb_model.pkl")