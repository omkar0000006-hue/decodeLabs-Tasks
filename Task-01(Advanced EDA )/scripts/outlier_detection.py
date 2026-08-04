import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "data/processed/missing_values_handled.csv"
OUTPUT_PATH = "data/processed/outliers_removed.csv"

os.makedirs("output/charts", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def load_data():
    return pd.read_csv(INPUT_PATH)


def detect_outliers_iqr(df, column):

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    return lower, upper, outliers


def remove_outliers(df, column):

    lower, upper, outliers = detect_outliers_iqr(df, column)

    print(f"\n{column}")
    print(f"Lower Limit : {lower:.2f}")
    print(f"Upper Limit : {upper:.2f}")
    print(f"Outliers : {len(outliers)}")

    cleaned_df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]

    return cleaned_df


def plot_boxplot(df, column):

    plt.figure(figsize=(7,5))

    sns.boxplot(x=df[column])

    plt.title(f"{column} Boxplot")

    plt.tight_layout()

    plt.savefig(f"output/charts/{column}_boxplot.png")

    plt.close()


def main():

    df = load_data()

    numeric_columns = [
        "Quantity",
        "UnitPrice",
        "ItemsInCart",
        "TotalPrice"
    ]

    for column in numeric_columns:
        plot_boxplot(df, column)

    cleaned_df = df.copy()

    for column in numeric_columns:
        cleaned_df = remove_outliers(cleaned_df, column)

    cleaned_df.to_csv(OUTPUT_PATH, index=False)

    print("\nOriginal Shape :", df.shape)
    print("Cleaned Shape :", cleaned_df.shape)

    print("\nOutlier-free dataset saved successfully.")


if __name__ == "__main__":
    main()