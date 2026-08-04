import streamlit as st


def sidebar_filters(df):

    st.sidebar.title("Dashboard Filters")

    years = sorted(df["Date"].dt.year.unique())

    selected_years = st.sidebar.multiselect(
        "Year",
        years,
        default=years
    )

    months = [
        "January","February","March","April",
        "May","June","July","August",
        "September","October","November","December"
    ]

    selected_months = st.sidebar.multiselect(
        "Month",
        months,
        default=months
    )

    products = sorted(df["Product"].unique())

    selected_products = st.sidebar.multiselect(
        "Product",
        products,
        default=products
    )

    payments = sorted(df["PaymentMethod"].unique())

    selected_payments = st.sidebar.multiselect(
        "Payment Method",
        payments,
        default=payments
    )

    status = sorted(df["OrderStatus"].unique())

    selected_status = st.sidebar.multiselect(
        "Order Status",
        status,
        default=status
    )

    referrals = sorted(df["ReferralSource"].unique())

    selected_referrals = st.sidebar.multiselect(
        "Referral Source",
        referrals,
        default=referrals
    )

    filtered = df[
        (df["Date"].dt.year.isin(selected_years))
        &
        (df["Month"].isin(selected_months))
        &
        (df["Product"].isin(selected_products))
        &
        (df["PaymentMethod"].isin(selected_payments))
        &
        (df["OrderStatus"].isin(selected_status))
        &
        (df["ReferralSource"].isin(selected_referrals))
    ]

    return filtered


def dataset_table(df):

    st.markdown("## Filtered Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )


def download_button(df):

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset",
        data=csv,
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )


def summary(df):

    st.markdown("## Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        len(df)
    )

    c2.metric(
        "Columns",
        len(df.columns)
    )

    c3.metric(
        "Products",
        df["Product"].nunique()
    )