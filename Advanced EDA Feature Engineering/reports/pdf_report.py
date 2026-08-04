import os
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet

INPUT_PATH = "data/processed/feature_engineered_data.csv"

CHARTS_FOLDER = "output/charts"

OUTPUT_FOLDER = "output/pdf"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_data():
    return pd.read_csv(INPUT_PATH)


def add_chart(elements, filename, title, styles):

    path = os.path.join(CHARTS_FOLDER, filename)

    if os.path.exists(path):

        elements.append(
            Paragraph(title, styles["Heading2"])
        )

        elements.append(
            Spacer(1, 10)
        )

        img = Image(path)

        img.drawWidth = 450

        img.drawHeight = 250

        elements.append(img)

        elements.append(
            Spacer(1, 20)
        )


def create_pdf(df):

    styles = getSampleStyleSheet()

    pdf = SimpleDocTemplate(

        os.path.join(

            OUTPUT_FOLDER,

            "Advanced_EDA_Report.pdf"

        )

    )

    elements = []

    elements.append(

        Paragraph(

            "Advanced EDA & Feature Engineering",

            styles["Title"]

        )

    )

    elements.append(

        Paragraph(

            "Professional Analytics Report",

            styles["Heading2"]

        )

    )

    elements.append(

        Spacer(1, 25)

    )

    elements.append(

        Paragraph(

            "<b>Dataset Summary</b>",

            styles["Heading1"]

        )

    )

    elements.append(

        Paragraph(

            f"Dataset Shape : {df.shape}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Total Revenue : ${df['TotalPrice'].sum():,.2f}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Average Order Value : ${df['TotalPrice'].mean():,.2f}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Customers : {df['CustomerID'].nunique()}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Products : {df['Product'].nunique()}",

            styles["BodyText"]

        )

    )

    elements.append(

        Spacer(1, 20)

    )

    elements.append(

        Paragraph(

            "<b>Business Insights</b>",

            styles["Heading1"]

        )

    )

    elements.append(

        Paragraph(

            f"Top Selling Product : {df['Product'].value_counts().idxmax()}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Most Used Payment Method : {df['PaymentMethod'].value_counts().idxmax()}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Most Common Referral Source : {df['ReferralSource'].value_counts().idxmax()}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"High Value Orders : {(df['HighValueOrder']=='Yes').sum()}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Orders Using Coupon : {(df['DiscountApplied']=='Yes').sum()}",

            styles["BodyText"]

        )

    )

    elements.append(PageBreak())

    add_chart(elements, "sales_distribution.png", "Sales Distribution", styles)

    add_chart(elements, "monthly_sales.png", "Monthly Sales", styles)

    add_chart(elements, "payment_method.png", "Payment Method", styles)

    add_chart(elements, "order_status.png", "Order Status", styles)

    add_chart(elements, "top_products.png", "Top Products", styles)

    add_chart(elements, "correlation_heatmap.png", "Correlation Heatmap", styles)

    elements.append(PageBreak())

    elements.append(

        Paragraph(

            "Conclusion",

            styles["Heading1"]

        )

    )

    elements.append(

        Paragraph(

            "This project demonstrates a complete end-to-end Exploratory Data Analysis workflow including data cleaning, feature engineering, statistical analysis, visualization and business insight generation. The generated findings help understand customer behavior, sales performance and business trends for better decision making.",

            styles["BodyText"]

        )

    )

    pdf.build(elements)

    print("\nPDF Generated Successfully")

    print(

        "Location : output/pdf/Advanced_EDA_Report.pdf"

    )


def main():

    df = load_data()

    create_pdf(df)


if __name__ == "__main__":

    main()