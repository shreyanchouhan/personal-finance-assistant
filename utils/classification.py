from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@lru_cache(maxsize=1)
def _train_classifier():
    df = pd.read_csv("data/expense_data.csv")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(df["description"].astype(str), df["category"].astype(str))
    return pipe


def predict_expense_category(description: str) -> str:
    if not description or not description.strip():
        return "Uncategorized"
    return _train_classifier().predict([description])[0]
