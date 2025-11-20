"""
Customer Churn Prediction Dashboard
====================================
Streamlit app for churn predictions using trained pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_pipeline():
    """Load the trained pipeline and model"""
    try:
        model_path = "model/churn_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ============================================================================
# PREPROCESSING FUNCTIONS (matching your pipeline)
# ============================================================================

def engineer_features(data):
    """Engineer features matching the pipeline"""
    df = pd.DataFrame([data])
    
    # Fix TotalCharges
    if 'TotalCharges' not in df.columns:
        df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Engineered features
    df['CLV'] = df['MonthlyCharges'] * df['tenure']
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Tenure Groups
    df['TenureGroup'] = pd.cut(df['tenure'], 
                           bins=[-1, 12, 24, 48, np.inf],
                           labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
    
    # Service count
    df['ServiceCount'] = 0
    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'InternetService', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col != 'InternetService':
            df['ServiceCount'] += (df[col] == 'Yes').astype(int)
    df['ServiceCount'] += (df['InternetService'] != 'No').astype(int)
    
    # Binary indicators
    df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
    df['HasPhone'] = (df['PhoneService'] == 'Yes').astype(int)
    df['HasPremiumServices'] = ((df['OnlineSecurity'] == 'Yes') | 
                                   (df['OnlineBackup'] == 'Yes') |
                                   (df['DeviceProtection'] == 'Yes') |
                                   (df['TechSupport'] == 'Yes')).astype(int)
    df['HasStreaming'] = ((df['StreamingTV'] == 'Yes') | 
                             (df['StreamingMovies'] == 'Yes')).astype(int)
    df['ContractRisk'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['PaymentRisk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)
    df['SeniorNoSupport'] = ((df['SeniorCitizen'] == 1) & 
                                (df['TechSupport'] == 'No')).astype(int)
    df['HighValueCustomer'] = (df['MonthlyCharges'] > 70).astype(int)
    df['ChargeTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    return df

def encode_features(df):
    """Encode features matching the pipeline"""
    df = df.copy()
    
    # Binary features encoding
    binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'PaperlessBilling']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 0, 
                                       'No phone service': 0, 
                                       'No': 0, 
                                       'Yes': 1, 
                                       'Male': 1, 
                                       'Female': 0
                                       })
    
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
    
    return df

def get_risk_level(probability):
    """Determine risk level"""
    if probability >= 0.7:
        return "HIGH RISK", "#d62728"
    elif probability >= 0.4:
        return "MEDIUM RISK", "#ff7f0e"
    else:
        return "LOW RISK", "#2ca02c"

def generate_recommendations(data):
    """Generate recommendations"""
    recommendations = []
    
    if data.get('Contract') == 'Month-to-month':
        recommendations.append("🎯 Offer long-term contract discount")
    
    if data.get('TechSupport') == 'No' and data.get('InternetService') != 'No':
        recommendations.append("💡 Provide tech support trial")
    
    if data.get('OnlineSecurity') == 'No' and data.get('InternetService') != 'No':
        recommendations.append("🔒 Offer online security package")
    
    if data.get('PaymentMethod') == 'Electronic check':
        recommendations.append("💳 Encourage automatic payment")
    
    if data.get('tenure', 0) < 6:
        recommendations.append("🎁 Provide loyalty incentive")
    
    if data.get('MonthlyCharges', 0) > 70:
        recommendations.append("💰 Offer package optimization")
    
    if not recommendations:
        recommendations.append("✅ Continue monitoring")
    
    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.title("📊 Customer Churn Prediction System")
    st.markdown("---")
    
    # Load model
    model_data = load_pipeline()
    
    if model_data is None:
        st.error("⚠️ Model not found!")
        st.info("""
        **Please train the model first:**
        ```bash
        python churn_prediction_pipeline.py
        ```
        This will create `model/churn_model.pkl`
        """)
        return
    
    model = model_data['model']
    pipeline = model_data['pipeline']
    feature_names = model_data['feature_names']
    all_cols = model_data['all_cols']
    
    # Sidebar
    st.sidebar.title("🎯 About")
    st.sidebar.info("Predicts customer churn risk with actionable recommendations")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** Logistic Regression")
    st.sidebar.markdown("**Features:** 40+ features")
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis"])
    
    # ========================================================================
    # TAB 1: SINGLE PREDICTION
    # ========================================================================
    with tab1:
        st.header("Customer Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Basic Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            
            st.subheader("💰 Billing")
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", 
                                          "Credit card (automatic)"])
        
        with col2:
            st.subheader("📞 Phone Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            
            st.subheader("🌐 Internet Services")
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            
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
        
        # Predict button
        if st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True):
            try:
                # Prepare input
                input_data = {
                    'gender': gender,
                    'SeniorCitizen': senior_citizen,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
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
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges
                }
                
                # Engineer features
                df_features = engineer_features(input_data)
                
                # Encode features
                df_encoded = encode_features(df_features)
                
                # Ensure correct columns
                for col in feature_names:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                
                df_encoded = df_encoded[feature_names]
                
                # Transform using pipeline
                X_processed = pipeline.transform(df_encoded)
                X_df = pd.DataFrame(X_processed, columns=all_cols)
                
                # Predict
                probability = model.predict_proba(X_df)[0][1]
                prediction = model.predict(X_df)[0]
                
                risk_level, color = get_risk_level(probability)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Churn Probability", f"{probability*100:.1f}%")
                
                with col2:
                    st.markdown(f'<div style="text-align: center;"><p style="font-size:2rem; color:{color}; font-weight:bold;">{risk_level}</p></div>', 
                              unsafe_allow_html=True)
                
                with col3:
                    st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score"},
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
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💡 Recommendations")
                    recommendations = generate_recommendations(input_data)
                    for rec in recommendations:
                        st.info(rec)
                
                with col2:
                    st.subheader("⚠️ Risk Factors")
                    risk_factors = []
                    
                    if contract == 'Month-to-month':
                        risk_factors.append("Month-to-month contract")
                    if tenure < 6:
                        risk_factors.append("New customer")
                    if payment_method == 'Electronic check':
                        risk_factors.append("Electronic check payment")
                    if tech_support == 'No' and internet_service != 'No':
                        risk_factors.append("No tech support")
                    if monthly_charges > 70:
                        risk_factors.append("High charges")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(f"⚠️ {factor}")
                    else:
                        st.success("✅ No major risks")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 2: BATCH ANALYSIS
    # ========================================================================
    with tab2:
        st.header("Batch Customer Analysis")
        
        st.info("📤 Upload CSV with customer data")
        
        # Sample format
        with st.expander("📋 Required CSV columns"):
            st.code("""
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges
            """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} customers")
                st.dataframe(df.head())
                
                if st.button("🚀 Analyze All", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        try:
                            # Engineer features
                            df_feat = engineer_features(row.to_dict())
                            
                            # Encode
                            df_enc = encode_features(df_feat)
                            
                            # Ensure columns
                            for col in feature_names:
                                if col not in df_enc.columns:
                                    df_enc[col] = 0
                            df_enc = df_enc[feature_names]
                            
                            # Transform
                            X_proc = pipeline.transform(df_enc)
                            X_df = pd.DataFrame(X_proc, columns=all_cols)
                            
                            # Predict
                            prob = model.predict_proba(X_df)[0][1]
                            pred = model.predict(X_df)[0]
                            
                            results.append({
                                'customerID': row.get('customerID', f'Customer_{idx}'),
                                'Churn_Probability': prob,
                                'Churn_Prediction': 'Yes' if pred == 1 else 'No',
                                'Risk_Level': 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'
                            })
                        except:
                            results.append({
                                'customerID': row.get('customerID', f'Customer_{idx}'),
                                'Churn_Probability': 0.5,
                                'Churn_Prediction': 'Unknown',
                                'Risk_Level': 'Unknown'
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary
                    st.subheader("📊 Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(results_df))
                    col2.metric("High Risk", (results_df['Risk_Level'] == 'High').sum())
                    col3.metric("Medium Risk", (results_df['Risk_Level'] == 'Medium').sum())
                    col4.metric("Low Risk", (results_df['Risk_Level'] == 'Low').sum())
                    
                    # Chart
                    fig = px.histogram(results_df, x='Churn_Probability', 
                                     title='Probability Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Results")
                    st.dataframe(results_df.sort_values('Churn_Probability', ascending=False))
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()