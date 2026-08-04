import streamlit as st


def format_currency(value):
    return f"${value:,.2f}"


def show_metrics(df):

    total_revenue = df["TotalPrice"].sum()

    total_orders = len(df)

    total_customers = df["CustomerID"].nunique()

    avg_order = df["TotalPrice"].mean()

    high_value_orders = (
        df["HighValueOrder"] == "Yes"
    ).sum()

    coupon_usage = (
        df["DiscountApplied"] == "Yes"
    ).mean() * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">💰 Revenue</div>
                <div class="metric-value">{format_currency(total_revenue)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">📦 Orders</div>
                <div class="metric-value">{total_orders:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">👥 Customers</div>
                <div class="metric-value">{total_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">🛒 Avg Order</div>
                <div class="metric-value">{format_currency(avg_order)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">⭐ High Value</div>
                <div class="metric-value">{high_value_orders:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">🏷 Coupon Usage</div>
                <div class="metric-value">{coupon_usage:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )