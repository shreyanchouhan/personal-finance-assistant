import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    return df.dropna().reset_index(drop=True)
