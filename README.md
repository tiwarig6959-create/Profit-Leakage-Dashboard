# Profit Leakage & Festive Growth Analysis
### Indian Retail Business Intelligence Dashboard

---

## Overview

A full-stack interactive Business Intelligence dashboard built entirely in **Python** that analyses 20,000 retail orders to uncover:

- **Profit Leakage** caused by excessive discounting
- **Festive Season Growth** patterns (Diwali, Navratri, Holi, Raksha Bandhan)
- **Regional & City-level** performance across India
- **Sales Channel & Payment** profitability
- **Customer Segment** insights by age, gender, and spending behaviour
- **ML Model Metrics** for profit prediction and leakage classification
- **Dataset** — `sales_data.csv` (20,000 orders · 17 columns · 17 Indian cities · 4 regions)
- **Backend** — `pandas`, `numpy`, `scikit-learn` (analytics functions + ML models)
- **Frontend** — `Dash`, `Plotly`, `Dash Bootstrap Components` (6 interactive tabs)

---

## Project Structure

```
Ibm project/
├── sales_data.csv                         <- Source data (20,000 orders)
│
└── profit_leakage_dashboard/
    ├── main_app.py                        <- SINGLE FILE: Analytics + ML + Dashboard
    ├── app.py                             <- Original multi-file frontend
    ├── analytics.py                       <- Analytics backend module
    ├── requirements.txt                   <- All Python dependencies
    ├── assets/
    │   └── style.css                      <- Dashboard theme (saffron palette)
    ├── models/
    │   ├── profit_regression.pkl          <- Trained RandomForest regressor
    │   ├── leakage_classifier.pkl         <- Trained GradientBoosting classifier
    │   └── leakage_scaler.pkl             <- StandardScaler for classifier
    └── screenshots/
        ├── screenshot_leakage.png
        ├── screenshot_festive.png
        ├── screenshot_regional.png
        ├── screenshot_channel.png
        └── screenshot_customer.png
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard (single file)
```bash
cd profit_leakage_dashboard
python main_app.py
```

### 3. Open your browser
```
http://127.0.0.1:8050
```

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| **Frontend UI** | `Dash` + `Dash Bootstrap Components` | Web framework, layout, cards |
| **Charts** | `Plotly` | All interactive visualisations |
| **Data** | `Pandas`, `NumPy` | Wrangling, aggregation, feature engineering |
| **ML** | `scikit-learn`, `joblib` | Model training, persistence |

---

## Dataset — `sales_data.csv`

| Column | Type | Description |
|---|---|---|
| `order_id` | str | Unique order identifier |
| `order_date` | date | Transaction date (2023–2025) |
| `customer_id` | str | Customer identifier |
| `gender` | str | Male / Female |
| `age` | int | Customer age |
| `region` | str | North / South / East / West |
| `city` | str | 17 Indian cities |
| `category` | str | Product category |
| `product` | str | Product name |
| `unit_price` | float | Price per unit (INR) |
| `quantity` | int | Units purchased |
| `discount` | float | Discount fraction (0.0–0.5) |
| `sales` | float | Net sales after discount |
| `profit` | float | Profit on order |
| `channel` | str | Store / Online / Mobile App |
| `payment_method` | str | UPI / Cash / Card / COD |
| `rating` | float | Customer rating (1–5) |

---

## Dashboard Tabs

### 1. Profit Leakage
- **Revenue Lost to Discounts** — total ₹ lost per category
- **Discount Band Analysis** — profit margin drops by discount tier
- **Top 15 Leaking Products** — ranked by revenue lost
- **Scatter: Discount% vs Profit Margin** — per-product bubble chart

### 2. Festive Growth
- **Monthly Sales & Profit Trend** with festive period shading
- **Festive vs Regular** avg order value comparison by category
- **Month-wise Profit Margin** with festive months highlighted
- KPIs: +4.91% festive sales uplift, Diwali peak detection

### 3. Regional Analysis
- **Sales & Profit by Region** grouped bar
- **Region Share Donut** chart
- **Top 20 Cities Treemap** coloured by profit margin
- **Leakage by Region** bar chart

### 4. Channel & Payment
- **Channel Profitability** (Store vs Online vs Mobile App)
- **Payment Method Distribution** donut
- **Margin vs Customer Rating** scatter by channel
- **Category × Channel Heatmap** of profit margins

### 5. Customer Insights
- **Age Group × Gender Sales** segmentation
- **Profit Margin Heatmap** by age and gender
- **Top 20 Customers** by lifetime revenue
- **Rating Distribution** heatmap by category

### 6. ML Model Metrics
- **Random Forest Profit Regressor** — MAE and R² on test set
- **Gradient Boosting Leakage Classifier** — Accuracy, Precision, Recall, F1
- **Feature Importance** bar chart
- Classification report visualisation

---

## ML Models

### Model 1 — Profit Regression (RandomForestRegressor)
```
Features : unit_price, quantity, discount_pct, sales, region,
           category, channel, month, age, is_festive
Target   : profit
Saved to : models/profit_regression.pkl
```

### Model 2 — Leakage Classifier (GradientBoostingClassifier)
```
Features : unit_price, quantity, sales, profit_margin,
           region, category, channel, month, age
Target   : is_high_discount (discount > 10%)
Saved to : models/leakage_classifier.pkl
           models/leakage_scaler.pkl
```

---

## Key Findings

| Insight | Value |
|---|---|
| Total Revenue Lost to Discounts | **Rs. 1.50 Cr** |
| Avg Profit Margin | **15.46%** |
| Festive Season Sales Uplift | **+4.91%** |
| High-Discount Orders (>10%) | **4,020 / 20,000** |
| Top Region by Sales | **South** (Rs. 7.01 Cr) |
| Best Margin Region | **North** (15.8%) |
| Total Cities Covered | **17** across 4 regions |

---

## Shared Filters

All 5 analytics tabs respond to the same filter bar:
- **Category** — filter by product category
- **Region** — filter by geographic region
- **Channel** — filter by sales channel
- **Date Range** — filter by custom date period

---

## Author

Built with Python, Dash, Plotly, Pandas, scikit-learn.
Indian Retail Business Intelligence — Profit Leakage & Festive Growth Analysis.
