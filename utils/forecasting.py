import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_expenses(forecast_data, months: int):
    series = forecast_data["expense"].astype(float).reset_index(drop=True)
    months = int(months)
    seasonal_periods = 12 if len(series) >= 24 else None
    if seasonal_periods:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=seasonal_periods
        ).fit()
    else:
        model = ExponentialSmoothing(series, trend="add").fit()
    forecast = model.forecast(months)
    return pd.Series(forecast.values, name="Forecasted Expense")
