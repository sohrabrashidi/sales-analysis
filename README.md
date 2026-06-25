# Sales Analysis

A data analysis pipeline for sales data with interactive Plotly visualizations and ML-based revenue forecasting.

## Features

- Automated data cleaning and preprocessing
- Interactive monthly revenue trend charts
- Category and regional breakdown visualizations
- 90-day revenue forecast using Ridge regression with seasonality features
- Self-contained HTML reports (no server needed)

## Tech Stack

| Tool | Purpose |
|------|---------|
| Pandas | Data loading, cleaning, aggregation |
| Plotly | Interactive HTML charts |
| Scikit-learn | Ridge regression for forecasting |
| NumPy | Numerical features (sin/cos seasonality) |

## Getting Started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/analyze.py
```

This generates a sample dataset and produces 4 HTML reports in `reports/`:

- `monthly_revenue.html` — Monthly revenue trend
- `category_breakdown.html` — Revenue by product category (donut chart)
- `regional_heatmap.html` — Revenue heatmap by region & month
- `forecast_90days.html` — 90-day revenue forecast

## Using Your Own Data

Replace `data/sales_data.csv` with your CSV. Required columns:

```
date, category, region, revenue, units_sold
```

## Project Structure

```
sales-analysis/
├── src/
│   ├── analyze.py    # Main pipeline
│   ├── clean.py      # Data cleaning functions
│   └── forecast.py   # ML forecasting
├── data/             # Place your CSV here
├── reports/          # Generated HTML reports
└── requirements.txt
```

## License

MIT
