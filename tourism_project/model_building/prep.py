import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns="ProdTaken"), df["ProdTaken"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train.to_csv(ROOT / "Xtrain.csv", index=False)
    X_test.to_csv(ROOT / "Xtest.csv", index=False)
    y_train.to_frame("ProdTaken").to_csv(ROOT / "ytrain.csv", index=False)
    y_test.to_frame("ProdTaken").to_csv(ROOT / "ytest.csv", index=False)
    print(f"Cleaned rows: {len(df):,}; predictors: {X.shape[1]}")
    print(f"Training rows: {len(X_train):,}; testing rows: {len(X_test):,}")
    print("Class balance (training):", y_train.value_counts(normalize=True).round(3).to_dict())

if __name__ == "__main__":
    main()
