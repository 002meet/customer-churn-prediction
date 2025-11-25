# Customer Churn Prediction System

A machine learning project that predicts customer churn for a telecom company and provides actionable recommendations to reduce customer loss.

🚀 **Live Demo:**  
https://customer-churn-prediction-hxwvsyq2yb4xczyjsfjniu.streamlit.app/

## Project Overview

This project analyzes customer data to predict which customers are likely to churn (leave the service). The model helps businesses identify at-risk customers early so they can take proactive retention measures.

**Key Results:**
- Model Accuracy: 84.7% ROC-AUC score
- Identifies 80% of customers who will churn
- Estimated annual savings: $400,000+
- Interactive web dashboard for real-time predictions

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
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset
│   └── churn_predictions.csv                    # Output predictions
│
├── model/
│   └── churn_model.pkl                          # Trained model
│
├── churn_prediction_pipeline.py                 # End-to-end ML pipeline
│                                   
├── app/
│    └──streamlit_app.py                         # Interactive web dashboard
│
├── requirements.txt                             # Dependencies
└── README.md                                    # Documentation
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
- Average Monthly Spend
- Service count
- Tenure groups
- Contract risk indicator
- New customer flag
- Premium service usage indicators

### 3. Data Preprocessing
- Handled missing values in TotalCharges column
- Converted categorical variables to numbers
- Applied one-hot encoding for multi-category features
- Split data into 80% training and 20% testing
- Scaled numerical features using StandardScaler

### 4. Handling Imbalanced Data
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Balanced the training data from 26.5% to 50% churn ratio
- This improved the model's ability to detect churners

### 5. Model Training
Trained and compared 6 different models:
- Logistic Regression ⭐
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

## Installation & Setup

### Requirements
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or download this repository**

2. **Create folder structure**
```bash
mkdir data model
```

3. **Install required packages**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn streamlit plotly
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Get it from [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` folder

5. **Train the model**
```bash
python churn_prediction_pipeline.py
```
This will:
- Process the data
- Train the model
- Save `model/churn_model.pkl`
- Generate `data/churn_predictions.csv`

6. **Launch the dashboard**
```bash
streamlit run app/streamlit_app.py
```
The app will open automatically at `http://localhost:8502`

## Using the Streamlit Dashboard

The interactive web dashboard provides powerful prediction capabilities with an intuitive interface.

### Features

#### 1. 🔍 Single Customer Prediction

**Real-time individual customer risk assessment:**

- **Input customer details** through interactive forms:
  - Basic information (gender, senior citizen status, partner, dependents, tenure)
  - Phone services (service type, multiple lines)
  - Internet services (type, security, backup, protection, tech support, streaming)
  - Billing information (contract type, payment method, charges)

- **Get instant predictions:**
  - Churn probability percentage (0-100%)
  - Risk level classification (Low/Medium/High)
  - Visual risk gauge with color coding
  - Predicted outcome (Will Churn / Will Stay)

- **Receive personalized recommendations:**
  - Contract upgrade suggestions for month-to-month customers
  - Tech support trial offers
  - Security package recommendations
  - Payment method improvements
  - Loyalty incentives for new customers
  - Package optimization for high-value customers

- **Identify specific risk factors:**
  - Month-to-month contract warning
  - New customer alert
  - Electronic check payment flag
  - Missing tech support indicator
  - High charges notification

**Example Use Case:**
A customer service representative can input a customer's details during a call and immediately see their churn risk, allowing for proactive retention offers.

#### 2. 📊 Batch Analysis

**Process multiple customers at once:**

- **Upload CSV file** with customer data
  - Supports any number of customers
  - Automatically validates required columns
  - Shows data preview before processing

- **Get comprehensive results:**
  - Churn probability for each customer
  - Risk level classification (High/Medium/Low)
  - Prediction (Yes/No) for each customer
  - Summary statistics dashboard

- **Visual analytics:**
  - Probability distribution histogram
  - Risk level breakdown
  - Total customer counts by risk category

- **Export results:**
  - Download predictions as CSV
  - Includes all customer IDs and risk scores
  - Ready for CRM integration

**Example Use Case:**
Marketing team can upload their entire customer database monthly to identify at-risk customers for targeted retention campaigns.

### Dashboard Navigation

**Sidebar:**
- Model information and performance metrics
- Quick about section
- Feature count display

**Main Interface:**
- Tab-based navigation for easy switching
- Responsive design works on all devices
- Clean, professional layout

### CSV Upload Format

For batch analysis, your CSV should include these columns:
```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges
```

### Dashboard Screenshots

**Single Prediction View:**
- Interactive form inputs
- Real-time risk gauge
- Color-coded risk levels (Green/Yellow/Red)
- Actionable recommendations list

**Batch Analysis View:**
- File upload interface
- Progress bar during processing
- Summary metrics cards
- Interactive probability distribution chart
- Sortable results table
- One-click CSV download

## Pipeline Architecture

The `churn_prediction_pipeline.py` file contains a complete, production-ready ML pipeline:

### Key Components:

1. **Data Loading & Cleaning**
   - Handles missing values
   - Fixes data type issues
   - Validates input data

2. **Feature Engineering**
   - Creates 14 engineered features
   - Calculates customer metrics
   - Generates risk indicators

3. **Preprocessing**
   - Encodes categorical variables
   - Applies one-hot encoding
   - Scales numerical features
   - Uses ColumnTransformer for proper pipeline

4. **Model Training**
   - SMOTE for class balancing
   - Logistic Regression classifier
   - Cross-validation
   - Performance evaluation

5. **Prediction & Export**
   - Processes new customer data
   - Generates risk scores
   - Exports to CSV

### Pipeline Usage:

**Training:**
```python
pipeline = ChurnPredictionPipeline("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
pipeline.fit()
pipeline.save_model("model/churn_model.pkl")
```

**Prediction:**
```python
pipeline = ChurnPredictionPipeline.load_model("model/churn_model.pkl")
predictions = pipeline.predict("data/new_customers.csv", "data/predictions.csv")
```

## Technologies Used

**Core Libraries:**
- **Python 3.8+** - Programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning framework
- **Imbalanced-learn (SMOTE)** - Handling imbalanced datasets

**Visualization:**
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive charts in dashboard

**Machine Learning Models:**
- **Logistic Regression** - Primary model
- **XGBoost & LightGBM** - Ensemble methods
- **Random Forest** - Tree-based ensemble

**Web Application:**
- **Streamlit** - Interactive web dashboard
- **Pickle** - Model serialization

**Development Tools:**
- **Jupyter Notebook** - Exploratory analysis
- **Git** - Version control

## Model Performance Details

### Confusion Matrix Analysis:
- True Negatives: Correctly predicted non-churners
- True Positives: Correctly identified churners
- False Positives: Unnecessary retention campaigns (low cost)
- False Negatives: Missed churners (high cost)

### ROC Curve:
- Area Under Curve (AUC): 84.7%
- Shows excellent discrimination ability
- Much better than random classifier (50%)

### Feature Importance:
Top 5 most important features for predictions:
1. Contract type (especially month-to-month)
2. Tenure (customer longevity)
3. Monthly charges
4. Tech support availability
5. Payment method

## Future Improvements

**Planned Enhancements:**
- Add time-series analysis to track churn trends over time
- Implement A/B testing framework for retention strategies
- Create REST API for real-time predictions
- Add automated model retraining pipeline
- Integrate with CRM systems (Salesforce, HubSpot)
- Add customer segmentation and clustering
- Develop email alert system for high-risk customers
- Create mobile-responsive dashboard version
- Add explainable AI (SHAP values) visualization
- Implement model monitoring and drift detection

## Project Highlights

What makes this project stand out:

1. **Real business problem** - Addresses actual telecom industry challenge
2. **End-to-end solution** - From raw data to deployed application
3. **Business-focused** - Calculated actual dollar impact ($400K+ savings)
4. **Production-ready** - Complete pipeline with error handling
5. **Interactive dashboard** - User-friendly Streamlit web application
6. **Well documented** - Clear explanations and code comments
7. **Scalable design** - Can handle single or batch predictions
8. **Actionable insights** - Provides specific recommendations, not just predictions

## Troubleshooting

**Common Issues:**

1. **Model file not found**
   - Run `python churn_prediction_pipeline.py` first
   - Ensure `model/churn_model.pkl` exists

2. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+)

3. **Streamlit not opening**
   - Try different port: `streamlit app/run streamlit_app.py --server.port 8502`
   - Check firewall settings

4. **CSV upload errors in dashboard**
   - Verify CSV has all required columns
   - Check for proper encoding (UTF-8)
   - Ensure no missing required fields

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset provided by IBM Sample Data Sets via Kaggle
- Built as a portfolio project to demonstrate data science skills
- Inspired by real-world telecom churn challenges

## Contact

For questions, suggestions, or collaboration:
- Create an issue on GitHub
- Email: patelmeet2406@gmail.com

---

**⭐ If you find this project helpful, please star it on GitHub!**

---

