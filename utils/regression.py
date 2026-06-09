from functools import lru_cache

import numpy as np
from sklearn.linear_model import LinearRegression


@lru_cache(maxsize=1)
def _train_regressor_from_df_signature(income_values_tuple, budget_values_tuple):
    X = np.array(income_values_tuple).reshape(-1, 1)
    y = np.array(budget_values_tuple)
    model = LinearRegression().fit(X, y)
    return model


def recommend_budget(income: float, budget_data) -> float:
    income_tuple = tuple(budget_data["income"].astype(float).tolist())
    budget_tuple = tuple(budget_data["recommended_budget"].astype(float).tolist())
    model = _train_regressor_from_df_signature(income_tuple, budget_tuple)
    pred = float(model.predict(np.array([[income]]))[0])
    return max(0.0, pred)
