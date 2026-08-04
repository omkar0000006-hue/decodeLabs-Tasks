import streamlit as st
import pandas as pd

from styles import load_css
from metrics import show_metrics

from charts import (
    revenue_by_product,
    monthly_revenue,
    payment_distribution,
    order_status,
    referral_source,
    top_customers,
    correlation_heatmap,
    coupon_analysis,
    high_value_orders,
    revenue_by_quarter
)

from components import (
    sidebar_filters,
    dataset_table,
    download_button,
    summary
)

st.set_page_config(
    page_title="Advanced EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

load_css()

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/processed/feature_engineered_data.csv"
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

filtered_df = sidebar_filters(df)

st.markdown(
    '<div class="title">📊 Advanced EDA & Feature Engineering Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Interactive Business Analytics Dashboard</div>',
    unsafe_allow_html=True
)

show_metrics(filtered_df)

st.markdown(
    '<div class="section">📈 Sales Analytics</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    revenue_by_product(filtered_df)

with c2:
    monthly_revenue(filtered_df)

c3, c4 = st.columns(2)

with c3:
    payment_distribution(filtered_df)

with c4:
    order_status(filtered_df)

c5, c6 = st.columns(2)

with c5:
    referral_source(filtered_df)

with c6:
    revenue_by_quarter(filtered_df)

c7, c8 = st.columns(2)

with c7:
    coupon_analysis(filtered_df)

with c8:
    high_value_orders(filtered_df)

st.markdown(
    '<div class="section">👥 Customer Analytics</div>',
    unsafe_allow_html=True
)

top_customers(filtered_df)

st.markdown(
    '<div class="section">📊 Correlation Analysis</div>',
    unsafe_allow_html=True
)

correlation_heatmap(filtered_df)

summary(filtered_df)

download_button(filtered_df)

dataset_table(filtered_df)