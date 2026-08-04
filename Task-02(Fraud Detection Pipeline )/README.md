#  Fraud Detection Pipeline using Machine Learning

An end-to-end Machine Learning based fraud detection system that identifies suspicious transactions using customer order behavior and transaction patterns.

The project includes data preprocessing, feature engineering, handling class imbalance using SMOTE, model comparison, hyperparameter tuning, model deployment using Flask API, and an interactive Streamlit dashboard.



#  Project Features

✅ Data preprocessing pipeline  
✅ Feature engineering  
✅ Handling imbalanced data using SMOTE  
✅ Multiple machine learning algorithms  
✅ Hyperparameter tuning using GridSearchCV  
✅ Model evaluation and comparison  
✅ Feature importance analysis  
✅ Fraud prediction API using Flask  
✅ Interactive dashboard using Streamlit  



#  Project Architecture

```
Dataset
   |
   ↓
Data Cleaning
   |
   ↓
Feature Engineering
   |
   ↓
SMOTE Balancing
   |
   ↓
Model Training
   |
   ↓
Gradient Boosting Model
   |
   ↓
 ┌───────────────┐
 |               |
 ↓               ↓

Flask API     Streamlit Dashboard

```



#  Technology Stack

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- Gradient Boosting
- Random Forest
- Logistic Regression
- Decision Tree
- KNN
- SVM

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn
- Plotly

## Deployment

- Flask
- Streamlit



#  Project Structure

```
Fraud Detection Pipeline

│
├── data
│
├── models
│   └── best_gradient_boosting.pkl
│
├── scripts
│   ├── 08_train_models.py
│   ├── 09_hyperparameter_tuning.py
│   ├── 10_evaluate_models.py
│   ├── 11_feature_importance.py
│   └── 12_predict.py
│
├── output
│   ├── charts
│   └── reports
│
├── 13_flask_api.py
│
├── 14_streamlit_dashboard.py
│
├── requirements.txt
│
└── README.md

```



#  Machine Learning Models Used

The following algorithms were evaluated:

| Model | Accuracy |
|---|---|
| Logistic Regression | 86.67% |
| Decision Tree | 92.92% |
| Random Forest | 93.33% |
| Gradient Boosting | 94.58% |
| KNN | 85.42% |
| SVM | 81.67% |



#  Best Model

## Gradient Boosting Classifier

Performance:

| Metric | Score |
|---|---|
| Accuracy | 94.58% |
| ROC-AUC | 96.86% |
| Recall | 83.33% |
| F1 Score | 60.61% |



#  Feature Importance

Top factors influencing fraud prediction:

| Feature | Importance |
|-|-|
| Quantity | 55.15% |
| Total Price | 21.38% |
| Payment Method | 7.42% |
| Items In Cart | 4.63% |
| Weekend Order | 2.04% |



#  Installation

Clone repository:

```
git clone <repository-url>
```

Navigate to project:

```
cd Fraud-Detection-Pipeline
```

Install dependencies:

```
pip install -r requirements.txt
```



#  Running the Application

## 1. Run Flask API

```
python 13_flask_api.py
```

API will run:

```
http://127.0.0.1:5000
```



## 2. Run Streamlit Dashboard

```
python -m streamlit run 14_streamlit_dashboard.py
```

Dashboard:

```
http://localhost:8501
```



#  API Example

Input:

```json
{
"Quantity":3,
"TotalPrice":1350,
"PaymentMethod":1
}
```

Output:

```json
{
"Prediction":"LEGITIMATE",
"Fraud Probability":0.0008
}
```



#  Dashboard Features

The Streamlit dashboard provides:

- Transaction prediction
- Model performance metrics
- Feature importance visualization
- Fraud risk probability



# Future Improvements

Possible enhancements:

- Real-time transaction streaming
- Cloud deployment
- Database integration
- Automated model retraining
- Deep learning based fraud detection


