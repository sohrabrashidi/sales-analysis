import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def build_features(monthly: pd.DataFrame) -> tuple:
    """Create time-based features for forecasting."""
    df = monthly.copy()
    df["t"] = np.arange(len(df))
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    X = df[["t", "sin_month", "cos_month"]].values
    y = df["total_revenue"].values
    return X, y, df


def forecast_next_n(monthly: pd.DataFrame, n: int = 90) -> pd.DataFrame:
    """Forecast next n days of revenue using Ridge regression on monthly data."""
    X, y, df = build_features(monthly)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    last_t = len(df)
    future_dates = pd.date_range(df["period"].iloc[-1], periods=n + 1, freq="D")[1:]
    future = pd.DataFrame({"date": future_dates})
    future["t"] = np.arange(last_t, last_t + n)
    future["month"] = future["date"].dt.month
    future["sin_month"] = np.sin(2 * np.pi * future["month"] / 12)
    future["cos_month"] = np.cos(2 * np.pi * future["month"] / 12)

    X_future = scaler.transform(future[["t", "sin_month", "cos_month"]].values)
    future["predicted_revenue"] = model.predict(X_future) / 30  # daily estimate

    return future[["date", "predicted_revenue"]]
