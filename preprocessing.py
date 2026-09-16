import pandas as pd
import numpy as np


TOP_N_CUISINES = 12  # number of explicit cuisine classes; rest -> "Other"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw restaurant dataset from CSV or XLSX."""
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def extract_primary_cuisine(cuisines: str) -> str:
    """Take the first cuisine listed as the restaurant's primary cuisine."""
    if pd.isna(cuisines):
        return np.nan
    return str(cuisines).split(",")[0].strip()


def clean_and_engineer(df: pd.DataFrame, top_n: int = TOP_N_CUISINES) -> pd.DataFrame:
   
    df = df.copy()

    # 1. Need a cuisine label to learn from
    df = df.dropna(subset=["Cuisines"]).reset_index(drop=True)

    # 2. Primary cuisine + how many cuisines are listed
    df["Cuisine Count"] = df["Cuisines"].apply(lambda x: len(str(x).split(",")))
    df["Primary Cuisine"] = df["Cuisines"].apply(extract_primary_cuisine)

    # 3. Collapse long tail into "Other"
    top_cuisines = df["Primary Cuisine"].value_counts().nlargest(top_n).index
    df["Cuisine Class"] = np.where(
        df["Primary Cuisine"].isin(top_cuisines), df["Primary Cuisine"], "Other"
    )

    # 4. Numeric cleaning
    # A handful of rows have Average Cost for two == 0, which is not a
    # real price -- treat it as missing and impute with the median for
    # that country (falls back to global median where a country has none).
    df["Average Cost for two"] = df["Average Cost for two"].replace(0, np.nan)
    df["Average Cost for two"] = df.groupby("Country Code")["Average Cost for two"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Average Cost for two"] = df["Average Cost for two"].fillna(
        df["Average Cost for two"].median()
    )

    for col in ["Longitude", "Latitude", "Votes", "Aggregate rating"]:
        df[col] = df[col].fillna(df[col].median())

    # 5. Categorical cleaning
    for col in ["City", "Currency", "Rating text"]:
        df[col] = df[col].fillna("Unknown")

    # City has 140+ distinct values, most represented by only a handful of
    # restaurants (this dataset is dominated by the Delhi NCR region).
    # One-hot encoding every city would mostly add noisy, near-unique
    # columns, so we bucket long-tail cities into "Other City" the same way
    # we bucketed long-tail cuisines.
    top_cities = df["City"].value_counts().nlargest(20).index
    df["City"] = np.where(df["City"].isin(top_cities), df["City"], "Other City")

    # 6. Encode boolean-like Yes/No columns
    binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    return df


FEATURE_COLUMNS_NUMERIC = [
    "Average Cost for two",
    "Price range",
    "Votes",
    "Aggregate rating",
    "Longitude",
    "Latitude",
    "Cuisine Count",
    "Has Table booking",
    "Has Online delivery",
]

FEATURE_COLUMNS_CATEGORICAL = ["City", "Currency"]

TARGET_COLUMN = "Cuisine Class"


def build_feature_frame(df: pd.DataFrame):
    """Return (X, y) with categoricals one-hot encoded."""
    X_numeric = df[FEATURE_COLUMNS_NUMERIC]
    X_categorical = pd.get_dummies(
        df[FEATURE_COLUMNS_CATEGORICAL], drop_first=False
    )
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df[TARGET_COLUMN]
    return X, y
