# Customer Churn Prediction System

A machine learning project that predicts customer churn for a telecom company and provides actionable recommendations to reduce customer loss.

## Project Overview

This project analyzes customer data to predict which customers are likely to churn (leave the service). The model helps businesses identify at-risk customers early so they can take proactive retention measures.

**Key Results:**
- Model Accuracy: 84.7% ROC-AUC score
- Identifies 80% of customers who will churn
- Estimated annual savings: $400,000+
- Interactive web dashboard for predictions

## Dataset

**Source:** Telco Customer Churn dataset from Kaggle

**Size:** 7,043 customer records

**Features:**
- Customer demographics (gender, age, dependents)
- Services subscribed (phone, internet, streaming)
- Account information (contract type, payment method, tenure)
- Billing (monthly charges, total charges)

**Target Variable:** Churn (Yes/No) - 26.5% of customers churned

## Project Structure

```
customer-churn-prediction/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
├── 01_eda_telco_churn.ipynb                # Data exploration
├── 02_feature_engineering.ipynb            # Feature creation
├── 03_modeling.ipynb                       # Model training
├── streamlit_app.py                        # Web dashboard
├── requirements.txt                        # Dependencies
└── README.md                               # Documentation
```

## Key Findings

**Main factors that cause customers to churn:**

1. Contract type - Month-to-month contracts have 42% churn vs 3% for two-year contracts
2. Tenure - New customers (less than 6 months) churn at 50% rate
3. Payment method - Electronic check users churn at 45%
4. Tech support - Customers without tech support churn 15% more
5. Monthly charges - Higher charges correlate with higher churn

## Methodology

### 1. Exploratory Data Analysis
- Analyzed patterns in customer behavior
- Identified key factors related to churn
- Created visualizations to understand the data

### 2. Feature Engineering
Created 14 new features including:
- Customer Lifetime Value (CLV)
- Service count
- Contract risk indicator
- New customer flag
- Premium service usage

### 3. Data Preprocessing
- Handled missing values in TotalCharges column
- Converted categorical variables to numbers
- Split data into 80% training and 20% testing
- Scaled numerical features

### 4. Handling Imbalanced Data
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Balanced the training data from 26.5% to 50% churn ratio
- This improved the model's ability to detect churners

### 5. Model Training
Trained and compared 6 different models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

**Best Model:** Logistic Regression with 84.7% ROC-AUC

### 6. Model Evaluation
- Precision: 68.9%
- Recall: 80.2%
- F1-Score: 74.1%
- ROC-AUC: 84.7%

**Why these metrics matter:**
- Recall (80.2%) means we catch 80% of customers who will churn
- Precision (68.9%) means 69% of our predictions are correct
- This is much better than random guessing (50%)

## Business Impact

**Assumptions:**
- Customer Lifetime Value: $2,000
- Retention campaign cost: $100 per customer
- Campaign success rate: 30%

**Results:**
- Correctly identifies 80% of churning customers
- Saves approximately 90 customers per quarter
- Annual revenue saved: $180,000
- Campaign costs: $40,000
- Net benefit: $140,000+ annually
- Return on investment: 350%

## Installation

### Requirements
- Python 3.8 or higher
- Jupyter Notebook

### Setup Steps

1. Clone or download this repository

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn streamlit plotly
```

3. Download the dataset from Kaggle and place it in the project folder

4. Run the notebooks in order:
   - 01_eda_telco_churn.ipynb
   - 02_feature_engineering.ipynb
   - 03_modeling.ipynb

5. Launch the dashboard:
```bash
streamlit run streamlit_app.py
```

## Using the Dashboard

The web dashboard provides three main features:

**1. Single Customer Prediction**
- Enter customer information
- Get instant churn risk score
- View personalized recommendations

**2. Batch Analysis**
- Upload CSV file with multiple customers
- Get predictions for all at once
- Download results

**3. Business Insights**
- View model performance metrics
- See feature importance
- Understand business impact

## Technologies Used

- **Python** - Programming language
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **XGBoost & LightGBM** - Advanced ML models
- **SMOTE** - Handling imbalanced data
- **Streamlit** - Web dashboard
- **Jupyter Notebook** - Development environment

## Future Improvements

- Add time-series analysis to track trends
- Implement A/B testing for retention strategies
- Create REST API for real-time predictions
- Add automated model retraining
- Integrate with CRM systems
- Add customer segmentation

## Project Highlights

What makes this project stand out:

1. **Real business problem** - Not just academic exercise
2. **End-to-end pipeline** - From data to deployed application
3. **Business focus** - Calculated actual dollar impact
4. **Production ready** - Working web application
5. **Well documented** - Clear explanations throughout

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset provided by IBM Sample Data Sets via Kaggle
- Built as a portfolio project to demonstrate data science skills
