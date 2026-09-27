import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
print("Loading dataset...")
df = pd.read_csv("Dataset1/phishing.csv")
print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
df = df.drop_duplicates()
print("After removing duplicates:")
print("Rows:", df.shape[0])
print("\nTarget column:")
print(df["Result"].value_counts())
X = df.drop("Result", axis=1)
y = df["Result"]
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
y = y.map({
    -1: 1,
    1: 0
})
if y.isna().any():
    print("ERROR: Unexpected values found in Result column.")
    print(df["Result"].unique())
    raise ValueError("Please check the Result labels.")
print("\nTarget conversion completed.")
print("Phishing:", (y == 1).sum())
print("Legitimate:", (y == 0).sum())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nData split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
os.makedirs("models", exist_ok=True)
joblib.dump(
    (
        X_train,
        X_test,
        y_train,
        y_test
    ),
    "models/data.pkl"
)
joblib.dump(
    scaler,
    "models/scaler.pkl"
)
print("\n================================")
print("PREPROCESSING COMPLETED!")
print("================================")

print("Saved:")
print("models/data.pkl")
print("models/scaler.pkl")