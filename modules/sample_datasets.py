import pandas as pd
from sklearn.datasets import load_iris


def get_iris_dataframe():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.rename(columns={"target": "species"}, inplace=True)
    return df


def get_housing_dataframe():
    data = {
        "Area_sqft": [850, 920, 1100, 1250, 1400, 1600, 1750, 2000, 2250, 2500, 2800, 3200],
        "Bedrooms": [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "Age_years": [25, 20, 18, 15, 12, 10, 8, 7, 6, 5, 4, 3],
        "Price_lakhs": [32, 38, 46, 52, 58, 65, 72, 82, 92, 105, 118, 136],
    }
    return pd.DataFrame(data)
