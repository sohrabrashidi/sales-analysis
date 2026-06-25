"""Main analysis script — run with: python src/analyze.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.clean import load_and_clean, monthly_summary, category_summary
from src.forecast import forecast_next_n

DATA_PATH = "data/sales_data.csv"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def generate_sample_data():
    """Generate a realistic sample dataset if none exists."""
    import numpy as np
    np.random.seed(42)
    n = 5000
    categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports"]
    regions = ["North", "South", "East", "West", "Central"]

    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="H").date,
        "category": np.random.choice(categories, n, p=[0.3, 0.25, 0.2, 0.1, 0.15]),
        "region": np.random.choice(regions, n),
        "revenue": np.random.exponential(scale=150, size=n).round(2),
        "units_sold": np.random.randint(1, 20, n),
    })
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated sample dataset: {n} rows → {DATA_PATH}")


def main():
    if not Path(DATA_PATH).exists():
        generate_sample_data()

    print("Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)
    monthly = monthly_summary(df)
    cats = category_summary(df)

    print(f"  {len(df):,} records | {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total revenue: ${df['revenue'].sum():,.0f}")

    # Monthly revenue trend
    fig1 = px.line(monthly, x="period", y="total_revenue",
                   title="Monthly Revenue Trend", markers=True,
                   labels={"period": "Month", "total_revenue": "Revenue ($)"})
    fig1.write_html(REPORTS_DIR / "monthly_revenue.html")

    # Category breakdown
    fig2 = px.pie(cats, names="category", values="total_revenue",
                  title="Revenue by Product Category", hole=0.4)
    fig2.write_html(REPORTS_DIR / "category_breakdown.html")

    # Regional heatmap
    region_month = df.groupby(["region", "month"])["revenue"].sum().reset_index()
    fig3 = px.density_heatmap(region_month, x="month", y="region", z="revenue",
                              title="Revenue Heatmap by Region & Month",
                              labels={"month": "Month", "region": "Region"})
    fig3.write_html(REPORTS_DIR / "regional_heatmap.html")

    # Forecast
    forecast = forecast_next_n(monthly, n=90)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=monthly["period"], y=monthly["total_revenue"] / 30,
                              name="Historical (daily avg)", mode="lines"))
    fig4.add_trace(go.Scatter(x=forecast["date"], y=forecast["predicted_revenue"],
                              name="Forecast", mode="lines", line=dict(dash="dash")))
    fig4.update_layout(title="90-Day Revenue Forecast", xaxis_title="Date", yaxis_title="Daily Revenue ($)")
    fig4.write_html(REPORTS_DIR / "forecast_90days.html")

    print("\nReports saved to reports/:")
    for f in sorted(REPORTS_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
