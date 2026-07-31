import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
}

def main():
    df = pd.read_csv(DATA_PATH)
    actual = set(df.columns) - {"Unnamed: 0"}
    missing, unexpected = EXPECTED_COLUMNS - actual, actual - EXPECTED_COLUMNS
    if missing:
        raise ValueError(f"Dataset validation failed. Missing columns: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Dataset validation failed. Unexpected columns: {sorted(unexpected)}")
    if df.empty:
        raise ValueError("Dataset validation failed: file contains no rows")
    if not set(df["ProdTaken"].dropna().unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must contain only 0 and 1")
    print("DATA REGISTRATION SUCCESSFUL")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}; usable columns: {len(actual)}")
    print("Target counts:", df["ProdTaken"].value_counts().sort_index().to_dict())
    print(f"Missing values: {int(df.isna().sum().sum())}")

if __name__ == "__main__":
    main()
