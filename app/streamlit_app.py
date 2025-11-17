"""
Customer Churn Prediction Dashboard
====================================
Interactive Streamlit app for predicting customer churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('../model/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../model/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, metadata
    except:
        return None, None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_input(data):
    """Preprocess user input to match training data format"""
    
    # Create feature dictionary
    features = {}
    
    # Basic features
    features['tenure'] = data['tenure']
    features['MonthlyCharges'] = data['MonthlyCharges']
    features['TotalCharges'] = data['tenure'] * data['MonthlyCharges']
    
    # Binary features
    features['SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
    features['Partner'] = 1 if data['Partner'] == 'Yes' else 0
    features['Dependents'] = 1 if data['Dependents'] == 'Yes' else 0
    features['PhoneService'] = 1 if data['PhoneService'] == 'Yes' else 0
    features['PaperlessBilling'] = 1 if data['PaperlessBilling'] == 'Yes' else 0
    
    # MultipleLines
    features['MultipleLines'] = 1 if data['MultipleLines'] == 'Yes' else 0
    
    # Internet services
    has_internet = data['InternetService'] != 'No'
    features['OnlineSecurity'] = 1 if data['OnlineSecurity'] == 'Yes' else 0
    features['OnlineBackup'] = 1 if data['OnlineBackup'] == 'Yes' else 0
    features['DeviceProtection'] = 1 if data['DeviceProtection'] == 'Yes' else 0
    features['TechSupport'] = 1 if data['TechSupport'] == 'Yes' else 0
    features['StreamingTV'] = 1 if data['StreamingTV'] == 'Yes' else 0
    features['StreamingMovies'] = 1 if data['StreamingMovies'] == 'Yes' else 0
    
    # Engineered features
    features['CLV'] = features['MonthlyCharges'] * features['tenure']
    features['AvgMonthlySpend'] = features['TotalCharges'] / (features['tenure'] + 1)
    
    # Service count
    service_count = (features['PhoneService'] + (1 if has_internet else 0) + 
                    features['OnlineSecurity'] + features['OnlineBackup'] +
                    features['DeviceProtection'] + features['TechSupport'] +
                    features['StreamingTV'] + features['StreamingMovies'])
    features['ServiceCount'] = service_count
    
    # Other features
    features['HasInternet'] = 1 if has_internet else 0
    features['HasPhone'] = features['PhoneService']
    features['HasPremiumServices'] = max(features['OnlineSecurity'], features['OnlineBackup'],
                                        features['DeviceProtection'], features['TechSupport'])
    features['HasStreaming'] = max(features['StreamingTV'], features['StreamingMovies'])
    features['ContractRisk'] = 1 if data['Contract'] == 'Month-to-month' else 0
    features['PaymentRisk'] = 1 if data['PaymentMethod'] == 'Electronic check' else 0
    features['IsNewCustomer'] = 1 if features['tenure'] < 6 else 0
    features['SeniorNoSupport'] = 1 if (features['SeniorCitizen'] == 1 and features['TechSupport'] == 0) else 0
    features['HighValueCustomer'] = 1 if features['MonthlyCharges'] > 70 else 0
    features['ChargeTenureRatio'] = features['MonthlyCharges'] / (features['tenure'] + 1)
    
    # One-hot encoded features
    features['InternetService_Fiber optic'] = 1 if data['InternetService'] == 'Fiber optic' else 0
    features['InternetService_No'] = 1 if data['InternetService'] == 'No' else 0
    features['Contract_One year'] = 1 if data['Contract'] == 'One year' else 0
    features['Contract_Two year'] = 1 if data['Contract'] == 'Two year' else 0
    features['PaymentMethod_Credit card (automatic)'] = 1 if data['PaymentMethod'] == 'Credit card (automatic)' else 0
    features['PaymentMethod_Electronic check'] = 1 if data['PaymentMethod'] == 'Electronic check' else 0
    features['PaymentMethod_Mailed check'] = 1 if data['PaymentMethod'] == 'Mailed check' else 0
    
    return pd.DataFrame([features])

def get_risk_level(probability):
    """Determine risk level based on churn probability"""
    if probability >= 0.7:
        return "HIGH RISK", "risk-high", "#d62728"
    elif probability >= 0.4:
        return "MEDIUM RISK", "risk-medium", "#ff7f0e"
    else:
        return "LOW RISK", "risk-low", "#2ca02c"

def generate_recommendations(probability, data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if data['Contract'] == 'Month-to-month':
        recommendations.append("🎯 Offer long-term contract discount (1 or 2 year)")
    
    if data['TechSupport'] == 'No' and data['InternetService'] != 'No':
        recommendations.append("💡 Provide complimentary tech support trial")
    
    if data['OnlineSecurity'] == 'No' and data['InternetService'] != 'No':
        recommendations.append("🔒 Offer online security package")
    
    if data['PaymentMethod'] == 'Electronic check':
        recommendations.append("💳 Encourage automatic payment method switch")
    
    if data['tenure'] < 6:
        recommendations.append("🎁 Provide new customer loyalty incentive")
    
    if data['MonthlyCharges'] > 70:
        recommendations.append("💰 Offer personalized package optimization")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Continue monitoring customer satisfaction")
    
    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">📊 Customer Churn Prediction System</p>', 
                unsafe_allow_html=True)
    
    # Load model
    model, metadata = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first.")
        return
    
    # Sidebar
    st.sidebar.header("🎯 Model Performance")
    if metadata:
        st.sidebar.metric("Model", metadata['model_name'])
        st.sidebar.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")
        st.sidebar.metric("Precision", f"{metadata['precision']:.3f}")
        st.sidebar.metric("Recall", f"{metadata['recall']:.3f}")
        st.sidebar.metric("Annual Savings", f"${metadata['net_benefit']:,.0f}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 This system predicts customer churn risk and provides actionable recommendations.")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Insights"])
    
    # ========================================================================
    # TAB 1: SINGLE PREDICTION
    # ========================================================================
    with tab1:
        st.header("Customer Churn Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Customer Information")
            
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
        with col2:
            st.subheader("📞 Services")
            
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            
            internet_service = st.selectbox("Internet Service", 
                                          ["No", "DSL", "Fiber optic"])
            
            if internet_service != "No":
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                online_security = online_backup = device_protection = "No"
                tech_support = streaming_tv = streaming_movies = "No"
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📄 Contract Details")
            contract = st.selectbox("Contract", 
                                   ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
        with col4:
            st.subheader("💳 Payment")
            payment_method = st.selectbox("Payment Method", 
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", 
                                          "Credit card (automatic)"])
        
        # Predict button
        if st.button("🎯 Predict Churn Risk", type="primary"):
            # Prepare input
            input_data = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method
            }
            
            # Preprocess
            features = preprocess_input(input_data)
            
            # Predict
            probability = model.predict_proba(features)[0][1]
            risk_level, risk_class, color = get_risk_level(probability)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{probability*100:.1f}%")
            
            with col2:
                st.markdown(f'<p style="font-size:2rem; color:{color}; font-weight:bold;">{risk_level}</p>', 
                          unsafe_allow_html=True)
            
            with col3:
                potential_loss = 2000  # CLV
                st.metric("Potential Loss", f"${potential_loss:,}")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            recommendations = generate_recommendations(probability, input_data)
            
            for rec in recommendations:
                st.info(rec)
            
            # Risk factors
            st.subheader("⚠️ Key Risk Factors")
            risk_factors = []
            
            if input_data['Contract'] == 'Month-to-month':
                risk_factors.append("Month-to-month contract (high risk)")
            if input_data['tenure'] < 6:
                risk_factors.append("New customer (< 6 months)")
            if input_data['PaymentMethod'] == 'Electronic check':
                risk_factors.append("Electronic check payment method")
            if input_data['TechSupport'] == 'No' and input_data['InternetService'] != 'No':
                risk_factors.append("No tech support with internet service")
            if monthly_charges > 70:
                risk_factors.append("High monthly charges")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.success("✅ No major risk factors identified")
    
    # ========================================================================
    # TAB 2: BATCH ANALYSIS
    # ========================================================================
    with tab2:
        st.header("Batch Customer Analysis")
        
        st.info("📤 Upload a CSV file with customer data for batch predictions")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            # Read data
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(batch_data)} customers")
            st.dataframe(batch_data.head())
            
            if st.button("🚀 Analyze All Customers"):
                # Add predictions
                probabilities = []
                
                for idx, row in batch_data.iterrows():
                    features = preprocess_input(row.to_dict())
                    prob = model.predict_proba(features)[0][1]
                    probabilities.append(prob)
                
                batch_data['Churn_Probability'] = probabilities
                batch_data['Risk_Level'] = batch_data['Churn_Probability'].apply(
                    lambda x: get_risk_level(x)[0]
                )
                
                # Summary
                st.subheader("📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                high_risk = (batch_data['Churn_Probability'] >= 0.7).sum()
                medium_risk = ((batch_data['Churn_Probability'] >= 0.4) & 
                              (batch_data['Churn_Probability'] < 0.7)).sum()
                low_risk = (batch_data['Churn_Probability'] < 0.4).sum()
                
                col1.metric("Total Customers", len(batch_data))
                col2.metric("High Risk", high_risk, delta=f"{high_risk/len(batch_data)*100:.1f}%")
                col3.metric("Medium Risk", medium_risk, delta=f"{medium_risk/len(batch_data)*100:.1f}%")
                col4.metric("Low Risk", low_risk, delta=f"{low_risk/len(batch_data)*100:.1f}%")
                
                # Distribution chart
                fig = px.histogram(batch_data, x='Churn_Probability', 
                                 title='Churn Probability Distribution',
                                 labels={'Churn_Probability': 'Churn Probability'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.subheader("📋 Detailed Results")
                st.dataframe(batch_data.sort_values('Churn_Probability', ascending=False))
                
                # Download
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # TAB 3: INSIGHTS
    # ========================================================================
    with tab3:
        st.header("📈 Business Insights & Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Performance")
            if metadata:
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Score': [0.80, metadata['precision'], metadata['recall'], 
                             metadata['f1_score'], metadata['roc_auc']]
                })
                
                fig = px.bar(metrics_df, x='Metric', y='Score', 
                           title='Model Performance Metrics')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 Business Impact")
            if metadata:
                impact_data = {
                    'Metric': ['Annual Savings', 'ROI'],
                    'Value': [f"${metadata['net_benefit']:,.0f}", f"{metadata['roi']:.1f}%"]
                }
                impact_df = pd.DataFrame(impact_data)
                st.dataframe(impact_df, hide_index=True)
                
                st.info(f"""
                **Assumptions:**
                - Customer Lifetime Value: $2,000
                - Retention Campaign Cost: $100/customer
                - Campaign Success Rate: 30%
                """)
        
        st.subheader("📚 About This System")
        st.markdown("""
        This Customer Churn Prediction System uses machine learning to:
        
        - **Predict** which customers are likely to churn
        - **Identify** key risk factors driving churn
        - **Recommend** personalized retention strategies
        - **Calculate** potential business impact
        
        **How to Use:**
        1. Enter customer information in the Single Prediction tab
        2. Click "Predict Churn Risk" to get results
        3. Review recommendations and take action
        4. For multiple customers, use the Batch Analysis tab
        
        **Risk Levels:**
        - 🟢 **Low Risk** (< 40%): Minimal intervention needed
        - 🟡 **Medium Risk** (40-70%): Monitor closely, consider proactive outreach
        - 🔴 **High Risk** (≥ 70%): Immediate action required
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()