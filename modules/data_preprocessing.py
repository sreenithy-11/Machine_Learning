import pandas as pd


def preprocess_dataframe(df):
    result = df.copy()
    result = result.dropna(how="all")
    return result


def remove_missing_values(df):
    return df.dropna()


def remove_duplicate_records(df):
    return df.drop_duplicates()


def basic_summary(df):
    return pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing values": df.isna().sum(),
        "Unique values": df.nunique(),
    })
