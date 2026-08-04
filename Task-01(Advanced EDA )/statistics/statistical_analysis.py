import os
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

INPUT_PATH = "data/processed/feature_engineered_data.csv"

os.makedirs("output/reports", exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_PATH)
    return df


def correlation_test(df):

    r, p = pearsonr(
        df["Quantity"],
        df["TotalPrice"]
    )

    return r, p


def normality_test(df):

    sample = df["TotalPrice"]

    if len(sample) > 500:
        sample = sample.sample(
            500,
            random_state=42
        )

    stat, p = shapiro(sample)

    return stat, p


def anova_test(df):

    groups = []

    for payment in df["PaymentMethod"].unique():

        groups.append(
            df[df["PaymentMethod"] == payment]["TotalPrice"]
        )

    stat, p = f_oneway(*groups)

    return stat, p


def chi_square_test(df):

    table = pd.crosstab(
        df["DiscountApplied"],
        df["OrderStatus"]
    )

    stat, p, dof, expected = chi2_contingency(table)

    return stat, p


def generate_report(df):

    report = []

    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 40)

    report.append(f"\nDataset Shape : {df.shape}")

    r, p = correlation_test(df)

    report.append("\nPEARSON CORRELATION")
    report.append(f"Correlation : {r:.4f}")
    report.append(f"P-Value : {p:.4f}")

    stat, p = normality_test(df)

    report.append("\nSHAPIRO-WILK NORMALITY TEST")
    report.append(f"Statistic : {stat:.4f}")
    report.append(f"P-Value : {p:.4f}")

    stat, p = anova_test(df)

    report.append("\nANOVA TEST")
    report.append(f"Statistic : {stat:.4f}")
    report.append(f"P-Value : {p:.4f}")

    stat, p = chi_square_test(df)

    report.append("\nCHI-SQUARE TEST")
    report.append(f"Statistic : {stat:.4f}")
    report.append(f"P-Value : {p:.4f}")

    if p < 0.05:
        report.append(
            "\nResult : Significant relationship found."
        )
    else:
        report.append(
            "\nResult : No significant relationship found."
        )

    with open(
        "output/reports/statistical_analysis.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(report))

    print("\n".join(report))

    print("\nReport saved successfully.")


def main():

    df = load_data()

    generate_report(df)


if __name__ == "__main__":
    main()