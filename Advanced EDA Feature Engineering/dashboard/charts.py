import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def revenue_by_product(df):

    fig, ax = plt.subplots(figsize=(8,5))

    revenue = (
        df.groupby("Product")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
    )

    sns.barplot(
        x=revenue.index,
        y=revenue.values,
        palette="Blues_r",
        ax=ax
    )

    ax.set_title("Revenue by Product")
    ax.set_xlabel("")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=30)

    st.pyplot(fig)


def monthly_revenue(df):

    monthly = (
        df.groupby("Month")["TotalPrice"]
        .sum()
        .reindex([
            "January","February","March","April",
            "May","June","July","August",
            "September","October","November","December"
        ])
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
        linewidth=3
    )

    ax.set_title("Monthly Revenue")
    ax.set_xlabel("")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=45)

    st.pyplot(fig)


def payment_distribution(df):

    fig, ax = plt.subplots(figsize=(6,6))

    df["PaymentMethod"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")
    ax.set_title("Payment Method Distribution")

    st.pyplot(fig)


def order_status(df):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=df,
        x="OrderStatus",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Order Status")

    plt.xticks(rotation=20)

    st.pyplot(fig)


def referral_source(df):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=df,
        x="ReferralSource",
        palette="Set2",
        ax=ax
    )

    ax.set_title("Referral Sources")

    plt.xticks(rotation=20)

    st.pyplot(fig)


def top_customers(df):

    top = (
        df.groupby("CustomerID")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        x=top.index,
        y=top.values,
        palette="rocket",
        ax=ax
    )

    ax.set_title("Top 10 Customers")

    plt.xticks(rotation=45)

    st.pyplot(fig)


def correlation_heatmap(df):

    corr = df.select_dtypes(include="number").corr()

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)


def coupon_analysis(df):

    fig, ax = plt.subplots(figsize=(6,5))

    sns.countplot(
        data=df,
        x="DiscountApplied",
        palette="coolwarm",
        ax=ax
    )

    ax.set_title("Coupon Usage")

    st.pyplot(fig)


def high_value_orders(df):

    fig, ax = plt.subplots(figsize=(6,5))

    sns.countplot(
        data=df,
        x="HighValueOrder",
        palette="crest",
        ax=ax
    )

    ax.set_title("High Value Orders")

    st.pyplot(fig)


def revenue_by_quarter(df):

    revenue = (
        df.groupby("Quarter")["TotalPrice"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(7,5))

    sns.barplot(
        x=revenue.index,
        y=revenue.values,
        palette="mako",
        ax=ax
    )

    ax.set_title("Revenue by Quarter")
    ax.set_ylabel("Revenue ($)")

    st.pyplot(fig)