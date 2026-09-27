from scipy.io import arff
import pandas as pd
data, meta = arff.loadarff("training dataset.arff")
df = pd.DataFrame(data)
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

df.to_csv("phishing.csv", index=False)
print("ARFF successfully converted to CSV!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())