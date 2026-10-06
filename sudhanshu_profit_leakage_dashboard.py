# -*- coding: utf-8 -*-
"""
=============================================================================
  Profit Leakage & Festive Growth Analysis  —  Indian Retail
=============================================================================
  SINGLE-FILE APPLICATION
  Sections:
    1. DATA ANALYTICS BACKEND   (pandas/numpy computations)
    2. ML TRAINING MODULE       (RandomForest + GradientBoosting)
    3. DASH FRONTEND            (layout, callbacks, charts)

  Run:
      python app.py

  Open browser at:  http://127.0.0.1:8050
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback

# ML
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report,
)

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — DATA ANALYTICS BACKEND
# =============================================================================

FESTIVE_MONTHS = {10: "Navratri/Dussehra", 11: "Diwali", 12: "Christmas/NYE",
                   1: "Makar Sankranti", 3: "Holi", 8: "Raksha Bandhan"}


def load_data(path: str) -> pd.DataFrame:
    """Load sales_data.csv and engineer all derived columns."""
    df = pd.read_csv(path, parse_dates=["order_date"])
    df["month"]         = df["order_date"].dt.month
    df["month_name"]    = df["order_date"].dt.strftime("%b")
    df["year"]          = df["order_date"].dt.year
    df["year_month"]    = df["order_date"].dt.to_period("M").astype(str)
    df["quarter"]       = df["order_date"].dt.to_period("Q").astype(str)
    df["profit_margin"] = np.where(df["sales"] > 0,
                                   (df["profit"] / df["sales"]) * 100, 0)
    df["discount_pct"]  = (df["discount"] * 100).round(1)
    df["revenue_lost"]  = (df["unit_price"] * df["quantity"] * df["discount"]).round(2)
    df["is_festive"]    = df["month"].isin(FESTIVE_MONTHS.keys())
    df["festive_label"] = df["month"].map(FESTIVE_MONTHS).fillna("Regular")
    df["discount_band"] = pd.cut(
        df["discount_pct"],
        bins=[-1, 0, 5, 10, 20, 100],
        labels=["0%", "1-5%", "6-10%", "11-20%", ">20%"]
    )
    return df


# ── Profit Leakage ────────────────────────────────────────────────────────────

def leakage_kpis(df: pd.DataFrame) -> dict:
    total_revenue_lost = df["revenue_lost"].sum()
    total_profit       = df["profit"].sum()
    total_sales        = df["sales"].sum()
    avg_margin         = (total_profit / total_sales * 100) if total_sales else 0
    high_discount_mask = df["discount_pct"] > 10
    leakage_ratio      = (total_revenue_lost / (total_sales + total_revenue_lost) * 100
                          if total_sales else 0)
    return {
        "total_revenue_lost":   round(total_revenue_lost, 0),
        "avg_profit_margin":    round(avg_margin, 2),
        "high_discount_orders": int(high_discount_mask.sum()),
        "leakage_ratio":        round(leakage_ratio, 2),
        "total_orders":         len(df),
    }


def discount_vs_margin(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("product", observed=True)
          .agg(avg_discount=("discount_pct", "mean"),
               avg_margin=("profit_margin", "mean"),
               total_sales=("sales", "sum"),
               order_count=("order_id", "count"))
          .reset_index().round(2)
    )


def leakage_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category", observed=True)
          .agg(revenue_lost=("revenue_lost", "sum"),
               total_profit=("profit", "sum"),
               total_sales=("sales", "sum"),
               avg_margin=("profit_margin", "mean"))
          .assign(margin=lambda x: (x["total_profit"] / x["total_sales"] * 100).round(2))
          .reset_index()
          .sort_values("revenue_lost", ascending=False)
    )


def discount_band_profit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("discount_band", observed=True)
          .agg(avg_margin=("profit_margin", "mean"),
               order_count=("order_id", "count"),
               total_profit=("profit", "sum"))
          .reset_index()
    )


def top_leaking_products(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        df[df["discount_pct"] > 0]
          .groupby("product", observed=True)
          .agg(revenue_lost=("revenue_lost", "sum"),
               avg_margin=("profit_margin", "mean"),
               avg_discount=("discount_pct", "mean"))
          .reset_index()
          .sort_values("revenue_lost", ascending=False)
          .head(n)
    )


# ── Festive Growth ────────────────────────────────────────────────────────────

def festive_kpis(df: pd.DataFrame) -> dict:
    fest = df[df["is_festive"]]
    reg  = df[~df["is_festive"]]
    fest_avg = fest["sales"].mean() if len(fest) else 0
    reg_avg  = reg["sales"].mean()  if len(reg)  else 0
    growth   = ((fest_avg - reg_avg) / reg_avg * 100) if reg_avg else 0
    return {
        "festive_avg_order":  round(fest_avg, 2),
        "regular_avg_order":  round(reg_avg, 2),
        "festive_growth_pct": round(growth, 2),
        "festive_margin":     round(fest["profit_margin"].mean() if len(fest) else 0, 2),
        "regular_margin":     round(reg["profit_margin"].mean()  if len(reg)  else 0, 2),
        "festive_orders":     len(fest),
    }


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["year", "month", "month_name"], observed=True)
          .agg(total_sales=("sales", "sum"),
               total_profit=("profit", "sum"),
               order_count=("order_id", "count"))
          .reset_index()
          .sort_values(["year", "month"])
    )
    monthly["period"]       = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    monthly["is_festive"]   = monthly["month"].isin(FESTIVE_MONTHS.keys())
    monthly["festive_label"] = monthly["month"].map(FESTIVE_MONTHS).fillna("")
    return monthly


def festive_vs_regular_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["category", "is_festive"], observed=True)
          .agg(avg_sales=("sales", "mean"),
               total_sales=("sales", "sum"),
               avg_margin=("profit_margin", "mean"))
          .reset_index()
    )


# ── Regional ──────────────────────────────────────────────────────────────────

def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region", observed=True)
          .agg(total_sales=("sales", "sum"),
               total_profit=("profit", "sum"),
               order_count=("order_id", "count"),
               avg_margin=("profit_margin", "mean"),
               revenue_lost=("revenue_lost", "sum"))
          .assign(margin=lambda x: (x["total_profit"] / x["total_sales"] * 100).round(2))
          .reset_index()
          .sort_values("total_sales", ascending=False)
    )


def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["region", "city"], observed=True)
          .agg(total_sales=("sales", "sum"),
               total_profit=("profit", "sum"),
               avg_margin=("profit_margin", "mean"))
          .reset_index()
          .sort_values("total_sales", ascending=False)
    )


# ── Channel & Payment ─────────────────────────────────────────────────────────

def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("channel", observed=True)
          .agg(total_sales=("sales", "sum"),
               total_profit=("profit", "sum"),
               avg_margin=("profit_margin", "mean"),
               order_count=("order_id", "count"),
               avg_rating=("rating", "mean"))
          .reset_index()
          .sort_values("total_sales", ascending=False)
    )


def payment_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("payment_method", observed=True)
          .agg(order_count=("order_id", "count"),
               total_sales=("sales", "sum"),
               avg_margin=("profit_margin", "mean"))
          .reset_index()
          .sort_values("order_count", ascending=False)
    )


def channel_category_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(index="category", columns="channel",
                       values="profit_margin", aggfunc="mean")
          .round(2)
    )


# ── Customer Insights ─────────────────────────────────────────────────────────

def customer_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 100],
                              labels=["≤25", "26-35", "36-45", "46-55", "55+"])
    return (
        df.groupby(["age_group", "gender"], observed=True)
          .agg(total_sales=("sales", "sum"),
               avg_margin=("profit_margin", "mean"),
               order_count=("order_id", "count"),
               avg_rating=("rating", "mean"))
          .reset_index()
    )


def top_customers(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return (
        df.groupby("customer_id", observed=True)
          .agg(total_sales=("sales", "sum"),
               total_profit=("profit", "sum"),
               order_count=("order_id", "count"),
               avg_discount=("discount_pct", "mean"))
          .reset_index()
          .sort_values("total_sales", ascending=False)
          .head(n)
    )


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["rating", "category"], observed=True)
          .agg(order_count=("order_id", "count"),
               avg_margin=("profit_margin", "mean"))
          .reset_index()
    )


# =============================================================================
# SECTION 2 — ML TRAINING MODULE
# =============================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

REGRESSION_FEATURES  = ["unit_price", "quantity", "discount_pct", "sales",
                         "region", "category", "channel", "month", "age", "is_festive"]
CLASSIFIER_FEATURES  = ["unit_price", "quantity", "sales", "profit_margin",
                         "region", "category", "channel", "month", "age"]


def _label_encode(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def train_profit_regression(df: pd.DataFrame) -> dict:
    """RandomForestRegressor → predicts order-level profit."""
    print("[1/2] Training Profit Regression (RandomForestRegressor)...")
    df_enc = _label_encode(df, ["region", "category", "channel"])
    df_enc["is_festive"] = df_enc["is_festive"].astype(int)

    X = df_enc[REGRESSION_FEATURES].fillna(0)
    y = df_enc["profit"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"  MAE={mae:.2f}  R²={r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "profit_regression.pkl")
    joblib.dump(model, out_path)
    print(f"  Saved → {out_path}")
    return {"mae": round(mae, 2), "r2": round(r2, 4), "features": REGRESSION_FEATURES}


def train_leakage_classifier(df: pd.DataFrame) -> dict:
    """GradientBoostingClassifier → identifies high-discount (>10%) orders."""
    print("[2/2] Training Leakage Classifier (GradientBoostingClassifier)...")
    df_enc = _label_encode(df, ["region", "category", "channel"])
    df_enc["is_high_discount"] = (df_enc["discount_pct"] > 10).astype(int)

    X = df_enc[CLASSIFIER_FEATURES].fillna(0)
    y = df_enc["is_high_discount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                       learning_rate=0.1, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Normal", "High Discount"],
                                   output_dict=True)
    print(f"  Accuracy={acc:.4f}  Precision(High)={report['High Discount']['precision']:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path  = os.path.join(MODEL_DIR, "leakage_classifier.pkl")
    scaler_path = os.path.join(MODEL_DIR, "leakage_scaler.pkl")
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved → {model_path}")
    print(f"  Saved → {scaler_path}")
    return {"accuracy": round(acc, 4), "features": CLASSIFIER_FEATURES, "report": report}


def train_all_models(df: pd.DataFrame) -> dict:
    print("\n>>> Training ML models on sales_data...")
    reg = train_profit_regression(df)
    cls = train_leakage_classifier(df)
    print(">>> Models saved to ./models/\n")
    return {"profit_regression": reg, "leakage_classifier": cls}


# =============================================================================
# SECTION 3 — DASH FRONTEND
# =============================================================================

# ── Colours (saffron-inspired Indian retail palette) ─────────────────────────
C_ORANGE  = "#FF6B35"
C_SAFFRON = "#FF9933"
C_GREEN   = "#138808"
C_BLUE    = "#1a73e8"
C_RED     = "#e53935"
C_PURPLE  = "#7c5cd8"
C_GOLD    = "#F5A623"
SEQ_ORANGE = px.colors.sequential.Oranges
DIV_RdGn   = px.colors.diverging.RdYlGn


def fmt_inr(n):
    if n >= 1e7: return f"₹{n/1e7:.2f} Cr"
    if n >= 1e5: return f"₹{n/1e5:.2f} L"
    if n >= 1e3: return f"₹{n/1e3:.1f} K"
    return f"₹{n:.0f}"


def kpi_card(title, value, subtitle="", color=C_ORANGE, icon=""):
    return dbc.Card([
        dbc.CardBody([
            html.P(f"{icon} {title}", className="kpi-title"),
            html.H3(value, className="kpi-value", style={"color": color}),
            html.P(subtitle, className="kpi-sub"),
        ])
    ], className="kpi-card")


def chart_layout(h=380):
    return dict(
        height=h, autosize=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#333"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font=dict(size=14, color="#222"),
        coloraxis_colorbar=dict(thickness=12, len=0.8),
    )


def graph_config():
    return {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "displaylogo": False,
        "responsive": False,
    }


# ── Load data & train models ──────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sales_data.csv")
df_raw    = load_data(DATA_PATH)
ml_metrics = train_all_models(df_raw)

CATEGORIES = sorted(df_raw["category"].unique())
REGIONS    = sorted(df_raw["region"].unique())
CHANNELS   = sorted(df_raw["channel"].unique())

# ── Dash app ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Profit Leakage & Festive Growth – Indian Retail",
    suppress_callback_exceptions=True,
)

# ── Shared filter bar ─────────────────────────────────────────────────────────
filter_bar = dbc.Card([
    dbc.Row([
        dbc.Col([
            html.Label("Category", className="filter-label"),
            dcc.Dropdown(
                id="filter-category",
                options=[{"label": "All Categories", "value": "ALL"}]
                        + [{"label": c, "value": c} for c in CATEGORIES],
                value="ALL", clearable=False, className="dash-dropdown",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Region", className="filter-label"),
            dcc.Dropdown(
                id="filter-region",
                options=[{"label": "All Regions", "value": "ALL"}]
                        + [{"label": r, "value": r} for r in REGIONS],
                value="ALL", clearable=False, className="dash-dropdown",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Channel", className="filter-label"),
            dcc.Dropdown(
                id="filter-channel",
                options=[{"label": "All Channels", "value": "ALL"}]
                        + [{"label": c, "value": c} for c in CHANNELS],
                value="ALL", clearable=False, className="dash-dropdown",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Date Range", className="filter-label"),
            dcc.DatePickerRange(
                id="filter-date",
                min_date_allowed=df_raw["order_date"].min(),
                max_date_allowed=df_raw["order_date"].max(),
                start_date=df_raw["order_date"].min(),
                end_date=df_raw["order_date"].max(),
                display_format="DD MMM YYYY",
                className="date-picker",
            ),
        ], md=3),
    ], className="g-2"),
], className="filter-bar mb-3")

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    html.Div([
        html.H1("🛒 Profit Leakage & Festive Growth Analysis",
                className="dashboard-title"),
        html.P("Indian Retail Business Intelligence Dashboard",
               className="dashboard-subtitle"),
    ], className="header-section"),
    filter_bar,
    dbc.Tabs([
        dbc.Tab(label="📉 Profit Leakage",    tab_id="tab-leakage"),
        dbc.Tab(label="🎉 Festive Growth",    tab_id="tab-festive"),
        dbc.Tab(label="🗺️ Regional Analysis", tab_id="tab-regional"),
        dbc.Tab(label="📦 Channel & Payment", tab_id="tab-channel"),
        dbc.Tab(label="👤 Customer Insights", tab_id="tab-customer"),
        dbc.Tab(label="🤖 ML Model Metrics",  tab_id="tab-ml"),
    ], id="tabs", active_tab="tab-leakage", className="main-tabs"),
    html.Div(id="tab-content", className="mt-3"),
], fluid=True, className="main-container")


# ── Shared filter helper ──────────────────────────────────────────────────────
def apply_filters(category, region, channel, start_date, end_date):
    df = df_raw.copy()
    if category != "ALL":
        df = df[df["category"] == category]
    if region != "ALL":
        df = df[df["region"] == region]
    if channel != "ALL":
        df = df[df["channel"] == channel]
    if start_date and end_date:
        df = df[(df["order_date"] >= start_date) & (df["order_date"] <= end_date)]
    return df


# ── Tab router callback ───────────────────────────────────────────────────────
@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("filter-category", "value"),
    Input("filter-region", "value"),
    Input("filter-channel", "value"),
    Input("filter-date", "start_date"),
    Input("filter-date", "end_date"),
)
def render_tab(active_tab, category, region, channel, start_date, end_date):
    df = apply_filters(category, region, channel, start_date, end_date)
    if active_tab == "tab-leakage":   return build_leakage_tab(df)
    if active_tab == "tab-festive":   return build_festive_tab(df)
    if active_tab == "tab-regional":  return build_regional_tab(df)
    if active_tab == "tab-channel":   return build_channel_tab(df)
    if active_tab == "tab-customer":  return build_customer_tab(df)
    if active_tab == "tab-ml":        return build_ml_tab()
    return html.Div("Select a tab")


# =============================================================================
# TAB 1 — PROFIT LEAKAGE
# =============================================================================
def build_leakage_tab(df):
    kpis  = leakage_kpis(df)
    cat   = leakage_by_category(df)
    disc  = discount_band_profit(df)
    top_l = top_leaking_products(df)
    scat  = discount_vs_margin(df)

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Revenue Lost to Discounts",
                         fmt_inr(kpis["total_revenue_lost"]),
                         f"Leakage ratio: {kpis['leakage_ratio']}%", C_RED, "💸"), md=3),
        dbc.Col(kpi_card("Avg Profit Margin",
                         f"{kpis['avg_profit_margin']}%",
                         f"Across {kpis['total_orders']:,} orders", C_ORANGE, "📊"), md=3),
        dbc.Col(kpi_card("High-Discount Orders (>10%)",
                         f"{kpis['high_discount_orders']:,}",
                         "Orders with discount > 10%", C_SAFFRON, "⚠️"), md=3),
        dbc.Col(kpi_card("Total Orders Analysed",
                         f"{kpis['total_orders']:,}",
                         "After applied filters", C_BLUE, "🛒"), md=3),
    ], className="g-3 mb-3")

    fig_cat = px.bar(cat, x="revenue_lost", y="category", orientation="h",
                     color="margin", color_continuous_scale=DIV_RdGn,
                     labels={"revenue_lost": "Revenue Lost (₹)", "category": "",
                             "margin": "Profit Margin %"},
                     title="Revenue Lost to Discounts by Category")
    fig_cat.update_layout(**chart_layout())
    fig_cat.update_traces(hovertemplate="<b>%{y}</b><br>Lost: ₹%{x:,.0f}<br>Margin: %{marker.color:.1f}%<extra></extra>")

    fig_band = px.bar(disc, x="discount_band", y="avg_margin",
                      color="avg_margin", color_continuous_scale=DIV_RdGn,
                      labels={"discount_band": "Discount Band", "avg_margin": "Avg Profit Margin %"},
                      title="Profit Margin vs Discount Band", text="avg_margin")
    fig_band.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_band.update_layout(**chart_layout())

    fig_top = px.bar(top_l, x="product", y="revenue_lost",
                     color="avg_discount", color_continuous_scale=SEQ_ORANGE,
                     labels={"product": "", "revenue_lost": "Revenue Lost (₹)",
                             "avg_discount": "Avg Discount %"},
                     title="Top 15 Products Causing Profit Leakage")
    fig_top.update_layout(**chart_layout(), xaxis_tickangle=-40)

    fig_scat = px.scatter(scat, x="avg_discount", y="avg_margin",
                          size="total_sales", color="avg_margin",
                          color_continuous_scale=DIV_RdGn, hover_name="product",
                          labels={"avg_discount": "Avg Discount %",
                                  "avg_margin": "Avg Profit Margin %",
                                  "total_sales": "Total Sales"},
                          title="Discount % vs Profit Margin per Product")
    fig_scat.add_hline(y=0, line_dash="dash", line_color=C_RED, annotation_text="Break-even")
    fig_scat.update_layout(**chart_layout())

    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cat,  config=graph_config()), md=6),
            dbc.Col(dcc.Graph(figure=fig_band, config=graph_config()), md=6),
        ], className="g-3 mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_top,  config=graph_config()), md=12)], className="g-3 mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_scat, config=graph_config()), md=12)], className="g-3"),
    ])


# =============================================================================
# TAB 2 — FESTIVE GROWTH
# =============================================================================
def build_festive_tab(df):
    kpis   = festive_kpis(df)
    trend  = monthly_sales_trend(df)
    fc_cat = festive_vs_regular_category(df)

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Festive Avg Order Value",
                         fmt_inr(kpis["festive_avg_order"]),
                         f"vs Regular: {fmt_inr(kpis['regular_avg_order'])}", C_SAFFRON, "🎉"), md=3),
        dbc.Col(kpi_card("Festive Sales Uplift",
                         f"+{kpis['festive_growth_pct']}%",
                         "vs regular months", C_GREEN, "📈"), md=3),
        dbc.Col(kpi_card("Festive Profit Margin",
                         f"{kpis['festive_margin']}%",
                         f"Regular: {kpis['regular_margin']}%", C_ORANGE, "💰"), md=3),
        dbc.Col(kpi_card("Festive Season Orders",
                         f"{kpis['festive_orders']:,}",
                         "Navratri · Diwali · Holi · Rakhi", C_PURPLE, "🛍️"), md=3),
    ], className="g-3 mb-3")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["period"], y=trend["total_sales"],
        mode="lines+markers", name="Total Sales",
        line=dict(color=C_ORANGE, width=2.5), marker=dict(size=6),
        hovertemplate="%{x}<br>Sales: ₹%{y:,.0f}<extra></extra>"
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend["period"], y=trend["total_profit"],
        mode="lines+markers", name="Total Profit",
        line=dict(color=C_GREEN, width=2, dash="dot"), marker=dict(size=5),
        hovertemplate="%{x}<br>Profit: ₹%{y:,.0f}<extra></extra>"
    ))
    for _, row in trend[trend["is_festive"]].iterrows():
        fig_trend.add_vrect(
            x0=row["period"], x1=row["period"],
            fillcolor=C_SAFFRON, opacity=0.15, layer="below", line_width=0,
            annotation_text=row["festive_label"],
            annotation_position="top left", annotation_font_size=9,
        )
    fig_trend.update_layout(**chart_layout(h=420),
                            title="Monthly Sales & Profit Trend (Festive Periods Highlighted)",
                            xaxis_tickangle=-45)

    fig_fc = px.bar(
        fc_cat, x="category", y="avg_sales",
        color=fc_cat["is_festive"].map({True: "Festive", False: "Regular"}),
        barmode="group",
        color_discrete_map={"Festive": C_SAFFRON, "Regular": C_BLUE},
        labels={"avg_sales": "Avg Order Value (₹)", "category": "", "color": "Period"},
        title="Festive vs Regular: Avg Order Value by Category",
    )
    fig_fc.update_layout(**chart_layout(), xaxis_tickangle=-30)

    month_margin = (
        df.groupby(["month_name", "month", "is_festive"], observed=True)
          .agg(avg_margin=("profit_margin", "mean"), total_sales=("sales", "sum"))
          .reset_index().sort_values("month")
    )
    fig_month = px.bar(
        month_margin, x="month_name", y="avg_margin",
        color=month_margin["is_festive"].map({True: "Festive", False: "Regular"}),
        color_discrete_map={"Festive": C_SAFFRON, "Regular": C_BLUE},
        labels={"month_name": "Month", "avg_margin": "Avg Profit Margin %"},
        title="Avg Profit Margin by Month (Festive Highlighted)",
    )
    fig_month.update_layout(**chart_layout())

    return html.Div([
        kpi_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_trend, config=graph_config()), md=12)], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_fc,    config=graph_config()), md=6),
            dbc.Col(dcc.Graph(figure=fig_month, config=graph_config()), md=6),
        ], className="g-3"),
    ])


# =============================================================================
# TAB 3 — REGIONAL ANALYSIS
# =============================================================================
def build_regional_tab(df):
    reg  = region_summary(df)
    city = city_summary(df)

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Top Region by Sales",
                         reg.iloc[0]["region"],
                         fmt_inr(reg.iloc[0]["total_sales"]), C_BLUE, "🥇"), md=3),
        dbc.Col(kpi_card("Best Margin Region",
                         reg.loc[reg["margin"].idxmax(), "region"],
                         f"{reg['margin'].max():.1f}% margin", C_GREEN, "💹"), md=3),
        dbc.Col(kpi_card("Most Leakage Region",
                         reg.loc[reg["revenue_lost"].idxmax(), "region"],
                         fmt_inr(reg["revenue_lost"].max()) + " lost", C_RED, "🚨"), md=3),
        dbc.Col(kpi_card("Total Cities",
                         str(df["city"].nunique()),
                         f"Across {df['region'].nunique()} regions", C_ORANGE, "🏙️"), md=3),
    ], className="g-3 mb-3")

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(x=reg["region"], y=reg["total_sales"],  name="Total Sales",  marker_color=C_BLUE))
    fig_reg.add_trace(go.Bar(x=reg["region"], y=reg["total_profit"], name="Total Profit", marker_color=C_GREEN))
    fig_reg.update_layout(**chart_layout(), barmode="group", title="Sales & Profit by Region")

    fig_donut = px.pie(reg, names="region", values="total_sales",
                       hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2,
                       title="Sales Distribution by Region")
    fig_donut.update_traces(textposition="outside", textinfo="percent+label",
                            hovertemplate="<b>%{label}</b><br>Sales: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>")
    fig_donut.update_layout(**chart_layout())

    fig_city = px.treemap(city.head(20), path=["region", "city"],
                          values="total_sales", color="avg_margin",
                          color_continuous_scale=DIV_RdGn,
                          labels={"total_sales": "Total Sales", "avg_margin": "Profit Margin %"},
                          title="Top 20 Cities by Sales (colour = Profit Margin)")
    fig_city.update_layout(**chart_layout(h=480))

    fig_leak = px.bar(reg, x="region", y="revenue_lost",
                      color="revenue_lost", color_continuous_scale=SEQ_ORANGE,
                      labels={"revenue_lost": "Revenue Lost (₹)", "region": ""},
                      title="Discount Revenue Lost by Region")
    fig_leak.update_layout(**chart_layout())

    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_reg,   config=graph_config()), md=8),
            dbc.Col(dcc.Graph(figure=fig_donut, config=graph_config()), md=4),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_city,  config=graph_config()), md=8),
            dbc.Col(dcc.Graph(figure=fig_leak,  config=graph_config()), md=4),
        ], className="g-3"),
    ])


# =============================================================================
# TAB 4 — CHANNEL & PAYMENT
# =============================================================================
def build_channel_tab(df):
    ch  = channel_summary(df)
    pay = payment_summary(df)
    hm  = channel_category_heatmap(df)

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Top Channel by Sales",
                         ch.iloc[0]["channel"],
                         fmt_inr(ch.iloc[0]["total_sales"]), C_BLUE, "🏆"), md=3),
        dbc.Col(kpi_card("Highest Margin Channel",
                         ch.loc[ch["avg_margin"].idxmax(), "channel"],
                         f"{ch['avg_margin'].max():.1f}% margin", C_GREEN, "📈"), md=3),
        dbc.Col(kpi_card("Top Payment Method",
                         pay.iloc[0]["payment_method"],
                         f"{pay.iloc[0]['order_count']:,} orders", C_ORANGE, "💳"), md=3),
        dbc.Col(kpi_card("Avg Customer Rating",
                         f"{df['rating'].mean():.2f} / 5.0",
                         "Across all channels", C_PURPLE, "⭐"), md=3),
    ], className="g-3 mb-3")

    fig_ch = go.Figure()
    fig_ch.add_trace(go.Bar(x=ch["channel"], y=ch["total_sales"],  name="Total Sales",  marker_color=C_BLUE))
    fig_ch.add_trace(go.Bar(x=ch["channel"], y=ch["total_profit"], name="Total Profit", marker_color=C_GREEN))
    fig_ch.update_layout(**chart_layout(), barmode="group", title="Sales & Profit by Channel")

    fig_chs = px.scatter(ch, x="avg_margin", y="avg_rating",
                         size="total_sales", color="channel",
                         labels={"avg_margin": "Avg Profit Margin %",
                                 "avg_rating": "Avg Customer Rating",
                                 "total_sales": "Total Sales"},
                         title="Channel: Margin vs Customer Satisfaction")
    fig_chs.update_layout(**chart_layout())

    fig_pay = px.pie(pay, names="payment_method", values="order_count",
                     hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel,
                     title="Order Share by Payment Method")
    fig_pay.update_traces(textinfo="percent+label")
    fig_pay.update_layout(**chart_layout())

    fig_hm = px.imshow(hm, color_continuous_scale=DIV_RdGn,
                       labels={"color": "Profit Margin %"},
                       title="Profit Margin Heatmap: Category × Channel",
                       aspect="auto", text_auto=".1f")
    fig_hm.update_layout(**chart_layout(h=420))

    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ch,  config=graph_config()), md=8),
            dbc.Col(dcc.Graph(figure=fig_pay, config=graph_config()), md=4),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_hm,  config=graph_config()), md=8),
            dbc.Col(dcc.Graph(figure=fig_chs, config=graph_config()), md=4),
        ], className="g-3"),
    ])


# =============================================================================
# TAB 5 — CUSTOMER INSIGHTS
# =============================================================================
def build_customer_tab(df):
    seg = customer_segment(df)
    top = top_customers(df, n=20)
    rat = rating_distribution(df)

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Unique Customers",
                         f"{df['customer_id'].nunique():,}",
                         "In current filter", C_BLUE, "👥"), md=3),
        dbc.Col(kpi_card("Avg Order Value",
                         fmt_inr(df["sales"].mean()),
                         "Per transaction", C_ORANGE, "🛒"), md=3),
        dbc.Col(kpi_card("Top Customer Revenue",
                         fmt_inr(top.iloc[0]["total_sales"]),
                         top.iloc[0]["customer_id"], C_GREEN, "🏅"), md=3),
        dbc.Col(kpi_card("Avg Customer Rating",
                         f"{df['rating'].mean():.2f} ⭐",
                         "Product satisfaction score", C_PURPLE, "💬"), md=3),
    ], className="g-3 mb-3")

    fig_seg = px.bar(seg, x="age_group", y="total_sales", color="gender",
                     barmode="group",
                     color_discrete_map={"Male": C_BLUE, "Female": C_SAFFRON},
                     labels={"age_group": "Age Group",
                             "total_sales": "Total Sales (₹)", "gender": "Gender"},
                     title="Sales by Age Group & Gender")
    fig_seg.update_layout(**chart_layout())

    seg_pivot = seg.pivot(index="age_group", columns="gender", values="avg_margin")
    fig_age_hm = px.imshow(seg_pivot, color_continuous_scale=DIV_RdGn,
                            labels={"color": "Avg Margin %"},
                            title="Profit Margin Heatmap: Age × Gender",
                            text_auto=".1f", aspect="auto")
    fig_age_hm.update_layout(**chart_layout(h=320))

    fig_top = px.bar(top, x="customer_id", y="total_sales",
                     color="avg_discount", color_continuous_scale=SEQ_ORANGE,
                     labels={"customer_id": "Customer ID",
                             "total_sales": "Total Sales (₹)",
                             "avg_discount": "Avg Discount %"},
                     title="Top 20 Customers by Revenue (colour = Avg Discount Given)")
    fig_top.update_layout(**chart_layout(), xaxis_tickangle=-45)

    rat_pivot = rat.pivot_table(index="category", columns="rating",
                                values="order_count", aggfunc="sum").fillna(0)
    fig_rat = px.imshow(rat_pivot, color_continuous_scale="Blues",
                        labels={"color": "Order Count", "x": "Rating", "y": "Category"},
                        title="Rating Distribution by Category",
                        text_auto=".0f", aspect="auto")
    fig_rat.update_layout(**chart_layout(h=380))

    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_seg,    config=graph_config()), md=8),
            dbc.Col(dcc.Graph(figure=fig_age_hm, config=graph_config()), md=4),
        ], className="g-3 mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_top, config=graph_config()), md=12)], className="g-3 mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_rat, config=graph_config()), md=12)], className="g-3"),
    ])


# =============================================================================
# TAB 6 — ML MODEL METRICS
# =============================================================================
def build_ml_tab():
    reg = ml_metrics["profit_regression"]
    cls = ml_metrics["leakage_classifier"]

    # Regression metrics cards
    reg_cards = dbc.Row([
        dbc.Col(kpi_card("Model",        "Random Forest Regressor", "Profit prediction",    C_BLUE,   "🌲"), md=3),
        dbc.Col(kpi_card("MAE",          f"₹{reg['mae']:.2f}",      "Mean Absolute Error",   C_ORANGE, "📏"), md=3),
        dbc.Col(kpi_card("R² Score",     f"{reg['r2']:.4f}",         "Variance explained",    C_GREEN,  "📐"), md=3),
        dbc.Col(kpi_card("Features Used", str(len(reg["features"])), "Input variables",       C_PURPLE, "🔢"), md=3),
    ], className="g-3 mb-3")

    # Classifier metrics cards
    report = cls["report"]
    cls_cards = dbc.Row([
        dbc.Col(kpi_card("Model",      "Gradient Boosting Classifier", "Leakage detection",            C_BLUE,   "🤖"), md=3),
        dbc.Col(kpi_card("Accuracy",   f"{cls['accuracy']:.4f}",        "Overall accuracy",             C_GREEN,  "✅"), md=3),
        dbc.Col(kpi_card("Precision",  f"{report['High Discount']['precision']:.4f}", "High Discount",  C_ORANGE, "🎯"), md=3),
        dbc.Col(kpi_card("F1 Score",   f"{report['High Discount']['f1-score']:.4f}",  "High Discount",  C_PURPLE, "⚖️"), md=3),
    ], className="g-3 mb-3")

    # Feature importance chart (regression model)
    feat_df = pd.DataFrame({
        "feature": reg["features"],
        "importance": joblib.load(os.path.join(MODEL_DIR, "profit_regression.pkl")).feature_importances_
    }).sort_values("importance", ascending=True)

    fig_feat = px.bar(feat_df, x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale=SEQ_ORANGE,
                      labels={"importance": "Feature Importance", "feature": ""},
                      title="Random Forest — Feature Importance (Profit Regression)")
    fig_feat.update_layout(**chart_layout(h=400))

    # Classification report bar chart
    classes = ["Normal", "High Discount"]
    metrics = ["precision", "recall", "f1-score"]
    fig_cls = go.Figure()
    colors  = [C_BLUE, C_SAFFRON, C_GREEN]
    for m, col in zip(metrics, colors):
        fig_cls.add_trace(go.Bar(
            name=m.capitalize(),
            x=classes,
            y=[report[c][m] for c in classes],
            marker_color=col,
            text=[f"{report[c][m]:.3f}" for c in classes],
            textposition="outside",
        ))
    fig_cls.update_layout(**chart_layout(), barmode="group",
                          title="Gradient Boosting — Classification Report",
                          yaxis_range=[0, 1.1])

    return html.Div([
        html.H5("Model 1 — Profit Regression", className="mt-2 mb-2"),
        reg_cards,
        html.H5("Model 2 — Leakage Classifier", className="mt-3 mb-2"),
        cls_cards,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_feat, config=graph_config()), md=6),
            dbc.Col(dcc.Graph(figure=fig_cls,  config=graph_config()), md=6),
        ], className="g-3 mt-2"),
    ])


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n>>>  Dashboard starting at  http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
