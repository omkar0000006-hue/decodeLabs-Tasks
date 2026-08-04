# 📊 Advanced EDA & Feature Engineering on E-Commerce Dataset

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-purple)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# 📌 Project Overview

This project was developed as part of the **DecodeLabs Data Science Internship**.

The objective of this project is to perform **Advanced Exploratory Data Analysis (EDA)** and **Feature Engineering** on an E-Commerce dataset to transform raw business data into a clean, structured, and analysis-ready dataset.

The project follows a complete data preprocessing workflow including:

- Data Loading
- Data Cleaning
- Missing Value Handling
- Outlier Detection & Treatment
- Feature Engineering
- Data Visualization
- Business Insight Generation

The processed dataset can be directly used for Machine Learning models, Business Intelligence dashboards, and Predictive Analytics.

---

# 🎯 Objectives

- Understand the structure of the dataset
- Clean and preprocess raw data
- Handle missing values efficiently
- Detect and remove outliers
- Engineer meaningful features
- Generate business insights
- Create professional visualizations
- Prepare data for future ML applications

---

# 📂 Dataset Information

**Dataset:** Synthetic E-Commerce Dataset

**Rows:** 1,200

**Columns:** 14

After preprocessing:

- Rows: **1,192**
- Columns: **20**

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📁 Project Structure

```
Advanced-EDA-Feature-Engineering/
│
├── data/
│   ├── raw/
│   │      ecommerce_data.csv
│   │
│   └── processed/
│          cleaned_data.csv
│          missing_values_handled.csv
│          outliers_removed.csv
│          feature_engineered_data.csv
│
├── notebooks/
│      EDA.ipynb
│
├── output/
│   ├── charts/
│   └── reports/
│
├── scripts/
│      load_data.py
│      data_cleaning.py
│      missing_values.py
│      outlier_detection.py
│      feature_engineering.py
│      visualization.py
│      generate_insights.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙ Features Implemented

## ✅ Data Loading

- Loaded dataset using Pandas
- Inspected dataset dimensions
- Checked column data types

---

## ✅ Data Cleaning

- Removed duplicate records
- Converted Date column into datetime format
- Standardized text columns
- Cleaned string values

---

## ✅ Missing Value Handling

- Missing value analysis
- Percentage calculation
- Filled missing CouponCode values using

```
No Coupon
```

---

## ✅ Outlier Detection

Applied IQR Method on:

- Quantity
- UnitPrice
- ItemsInCart
- TotalPrice

Detected and removed extreme values.

---

## ✅ Feature Engineering

Created six new business features:

| Feature | Description |
|----------|-------------|
| Month | Month extracted from Date |
| DayOfWeek | Day extracted from Date |
| Quarter | Financial Quarter |
| AverageItemPrice | TotalPrice / Quantity |
| DiscountApplied | Whether coupon was used |
| HighValueOrder | Order above median revenue |

---

## ✅ Data Visualization

Generated professional charts including:

- Revenue by Product
- Monthly Revenue
- Payment Method Distribution
- Order Status Distribution
- Referral Source Distribution
- Quantity Distribution
- Total Price Distribution
- Correlation Heatmap
- Discount Analysis
- High Value Orders

---

## ✅ Business Insights

Automatically generated insights such as:

- Total Revenue
- Average Order Value
- Total Customers
- Top Selling Product
- Most Used Payment Method
- Referral Source Analysis
- Order Status Distribution
- High Value Orders
- Coupon Usage

---

# 📈 Project Results

### Dataset Summary

| Metric | Value |
|---------|------:|
| Original Records | 1,200 |
| Final Records | 1,192 |
| Original Features | 14 |
| Final Features | 20 |

---

### Revenue Statistics

| Metric | Value |
|---------|--------:|
| Total Revenue | $1,237,728.56 |
| Average Order Value | $1,038.36 |
| Highest Order | $3,322.55 |
| Lowest Order | $11.39 |

---

### Business Insights

- Top Selling Product: **Printer**
- Most Used Payment Method: **Online**
- Most Common Referral Source: **Instagram**
- High Value Orders: **596**
- Orders Using Coupons: **885**
- Orders Without Coupons: **307**

---

# 📷 Screenshots

Create a folder named:

```
screenshots/
```

Add images such as:

```
screenshots/

dashboard.png

correlation_heatmap.png

monthly_revenue.png

sales_by_product.png

payment_method_distribution.png

order_status_distribution.png

high_value_orders.png
```

Then display them like this:

```markdown
## Dashboard

![Dashboard](screenshots/dashboard.png)

## Revenue by Product

![Revenue](screenshots/sales_by_product.png)

## Correlation Heatmap

![Heatmap](screenshots/correlation_heatmap.png)
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/omkar0000006-hue/Advanced-EDA-Feature-Engineering.git
```

Move into the project

```bash
cd Advanced-EDA-Feature-Engineering
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run scripts

```bash
python scripts/load_data.py

python scripts/data_cleaning.py

python scripts/missing_values.py

python scripts/outlier_detection.py

python scripts/feature_engineering.py

python scripts/visualization.py

python scripts/generate_insights.py
```

---

# 📊 Output

The project automatically generates

```
Processed datasets

Business reports

Visualizations

Feature engineered dataset

Business insights report
```

---

# 📚 Learning Outcomes

Through this project, I learned

- Data preprocessing
- Feature Engineering
- Outlier Detection
- Exploratory Data Analysis
- Business Analytics
- Data Visualization
- Python Data Analysis Libraries
- Professional Project Structuring
- Report Generation

---

# 🔮 Future Enhancements

- Build an interactive Streamlit Dashboard
- Add Power BI Dashboard
- Apply Machine Learning models
- Customer Segmentation
- Sales Forecasting
- Recommendation System
- Deploy on Streamlit Cloud

---

# 👨‍💻 Author

**Omkar Yeram**

Computer Engineering Student

Aspiring Data Engineer | Data Analyst

GitHub

https://github.com/omkar0000006-hue

LinkedIn

https://www.linkedin.com/in/omkar-yeram-803592377/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates further improvements.

---

## 📜 License

This project is licensed under the MIT License.