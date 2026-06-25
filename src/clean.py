import pandas as pd
import numpy as np


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV and apply cleaning pipeline."""
    df = pd.read_csv(path, parse_dates=["date"])

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows missing critical fields
    df = df.dropna(subset=["date", "revenue", "category"])

    # Fix negatives (returns logged as negative revenue)
    df["revenue"] = df["revenue"].abs()

    # Normalize category names
    df["category"] = df["category"].str.strip().str.title()

    # Extract time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    return df.sort_values("date").reset_index(drop=True)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by month."""
    return (
        df.groupby(["year", "month"])["revenue"]
        .agg(total_revenue="sum", num_orders="count", avg_order="mean")
        .reset_index()
        .assign(period=lambda x: pd.to_datetime(dict(year=x.year, month=x.month, day=1)))
        .sort_values("period")
    )


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by product category."""
    return (
        df.groupby("category")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
    )
