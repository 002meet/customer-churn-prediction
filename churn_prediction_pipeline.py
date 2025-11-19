# Customer Churn Prediction Pipeline
# =============================================

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    """Complete pipeline for customer churn prediction"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.all_cols = None
        
    def load_data(self):
        print("Step 1/6: Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Fix TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        
        print(f"   Loaded {df.shape[0]} customers")
        return df
    
    def clean_data(self, df):
        """Clean and fix data types"""
        df = df.copy()
        # Fix TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        return df
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        print("Step 2/6: Engineering features...")
        
        df_fe = df.copy()
        
        # Engineered features
        df_fe['CLV'] = df_fe['MonthlyCharges'] * df_fe['tenure']

        df_fe['TotalCharges'] = pd.to_numeric(df_fe['TotalCharges'], errors='coerce')
        df_fe['TotalCharges'].fillna(0, inplace=True)

        df_fe['AvgMonthlySpend'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)
        # 3. Tenure Groups
        df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], 
                               bins=[-1, 12, 24, 48, np.inf],
                               labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
        
        # Service count
        df_fe['ServiceCount'] = 0
        service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'InternetService', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in service_cols:
            if col != 'InternetService':
                df_fe['ServiceCount'] += (df_fe[col] == 'Yes').astype(int)
        df_fe['ServiceCount'] += (df_fe['InternetService'] != 'No').astype(int)
        
        # Binary indicators
        df_fe['HasInternet'] = (df_fe['InternetService'] != 'No').astype(int)
        df_fe['HasPhone'] = (df_fe['PhoneService'] == 'Yes').astype(int)
        df_fe['HasPremiumServices'] = ((df_fe['OnlineSecurity'] == 'Yes') | 
                                       (df_fe['OnlineBackup'] == 'Yes') |
                                       (df_fe['DeviceProtection'] == 'Yes') |
                                       (df_fe['TechSupport'] == 'Yes')).astype(int)
        df_fe['HasStreaming'] = ((df_fe['StreamingTV'] == 'Yes') | 
                                 (df_fe['StreamingMovies'] == 'Yes')).astype(int)
        df_fe['ContractRisk'] = (df_fe['Contract'] == 'Month-to-month').astype(int)
        df_fe['PaymentRisk'] = (df_fe['PaymentMethod'] == 'Electronic check').astype(int)
        df_fe['IsNewCustomer'] = (df_fe['tenure'] < 6).astype(int)
        df_fe['SeniorNoSupport'] = ((df_fe['SeniorCitizen'] == 1) & 
                                    (df_fe['TechSupport'] == 'No')).astype(int)
        df_fe['HighValueCustomer'] = (df_fe['MonthlyCharges'] > 
                                      df_fe['MonthlyCharges'].quantile(0.75)).astype(int)
        df_fe['ChargeTenureRatio'] = df_fe['MonthlyCharges'] / (df_fe['tenure'] + 1)
        
        print(f"   Created 14 new features")
        return df_fe
    
    def encode_features(self, df):
        """Encode categorical variables"""
        print("Step 3/6: Encoding categorical features...")
        
        df_enc = df.copy()
        
        # Encode target if present
        if 'Churn' in df_enc.columns:
            df_enc['Churn'] = df_enc['Churn'].map({'Yes': 1, 'No': 0})
        
        # Binary features
        binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'PaperlessBilling']
       
        for col in binary_cols:
            if col in df_enc.columns:
                df_enc[col] = df_enc[col].replace({'No internet service': 0, 
                                                   'No phone service': 0, 
                                                   'No': 0, 
                                                   'Yes': 1, 
                                                   'Male': 1, 
                                                   'Female': 0
                                                   })
        
        # Drop non-numeric columns except those for one-hot
        df_enc = df_enc.drop(['customerID'], axis=1, errors='ignore')
        
        print(f"   Final feature count: {df_enc.shape[1] - ('Churn' in df_enc.columns)}")
        return df_enc
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
        print("Step 4/6: Training Logistic Regression model...")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"   Model trained - ROC-AUC: {roc_auc:.3f}")
        
        return self.model
    
    def fit(self):
        """Complete training pipeline"""
        # Load data
        df = self.load_data()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode features
        df = self.encode_features(df)
        
        # Split features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Define columns
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                         'CLV', 'AvgMonthlySpend', 'ChargeTenureRatio']
        
        categorical_cols = ['InternetService', 'Contract', 
                            'PaymentMethod', 'TenureGroup']
        
        # One-hot encoding & Scaling
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        numeric_transformer = StandardScaler()

        # Combining everything
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Creating Pipeline
        self.pipeline = Pipeline(steps=[('preprocess', preprocessor)])
        
        # Fit-Transform on Training set & Transform on Test set
        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)

        # Get column names
        ohe_cols = self.pipeline.named_steps['preprocess'] \
                   .named_transformers_['cat'] \
                   .get_feature_names_out(categorical_cols)
        pass_cols = [col for col in X_train.columns if col not in categorical_cols + numerical_cols]
        self.all_cols = np.concatenate([pass_cols, numerical_cols, ohe_cols])
        
        X_train_df = pd.DataFrame(X_train_processed, columns=self.all_cols)
        X_test_df = pd.DataFrame(X_test_processed, columns=self.all_cols)

        # Train model
        self.train_model(X_train_df, y_train, X_test_df, y_test)
        
        # Final evaluation
        print("\nStep 5/6: Model Evaluation")
        y_pred = self.model.predict(X_test_df)
        y_pred_proba = self.model.predict_proba(X_test_df)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
        
        return X_test_df, y_test, y_pred, y_pred_proba
    
    def preprocess_new_data(self, df):
        """Preprocess new data for prediction"""
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode features
        df = self.encode_features(df)
        
        # Remove target if exists
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns
        df = df[self.feature_names]
        
        # Transform using pipeline
        X_new_processed = self.pipeline.transform(df)
        X_new_df = pd.DataFrame(X_new_processed, columns=self.all_cols)
        
        return X_new_df
    
    def predict(self, data_path, output_path='data/predictions.csv'):
        """Make predictions on new data and save to CSV"""
        print("\nStep 6/6: Making predictions...")
        
        # Load new data
        df_new = pd.read_csv(data_path)
        customer_ids = df_new['customerID'].copy()
        
        # Preprocess
        X_new = self.preprocess_new_data(df_new)
        
        # Predict
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'customerID': customer_ids,
            'Churn_Prediction': ['Yes' if p == 1 else 'No' for p in predictions],
            'Churn_Probability': probabilities,
            'Risk_Level': ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' 
                          for p in probabilities]
        })
        
        # Save to CSV
        results.to_csv(output_path, index=False)
        print(f"   Predictions saved to '{output_path}'")
        print(f"   Total customers: {len(results)}")
        print(f"   High Risk: {(results['Risk_Level'] == 'High').sum()}")
        print(f"   Medium Risk: {(results['Risk_Level'] == 'Medium').sum()}")
        print(f"   Low Risk: {(results['Risk_Level'] == 'Low').sum()}")
        
        return results
    
    def save_model(self, filepath="model/churn_model.pkl"):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'all_cols': self.all_cols
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved to '{filepath}'")
    
    @classmethod
    def load_model(cls, filepath="model/churn_model.pkl"):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        pipeline = cls.__new__(cls)
        pipeline.model = model_data['model']
        pipeline.pipeline = model_data['pipeline']
        pipeline.feature_names = model_data['feature_names']
        pipeline.all_cols = model_data['all_cols']
        
        return pipeline

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = pipeline.fit()
    
    # Save model
    pipeline.save_model("model/churn_model.pkl")
    
    # Make predictions on same data (for demonstration)
    print("\n" + "="*70)
    predictions = pipeline.predict("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", 
                                   "data/churn_predictions.csv")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput files created:")
    print("  1. churn_model.pkl - Trained model")
    print("  2. churn_predictions.csv - Predictions with probabilities")
    print("\nSample predictions:")
    print(predictions.head(10))
