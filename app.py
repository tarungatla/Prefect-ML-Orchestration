import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="wide",
)

# --- MODEL AND SCALER LOADING ---
# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model and scaler from disk.
    """
    model_dir = Path("models")
    model_path = model_dir / "best_model.pkl" # Assumes you've saved the best model with this name
    scaler_path = model_dir / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        st.error("Model or scaler not found! Please run the training pipeline first.")
        st.stop()
        
    with open(model_path, 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open(scaler_path, 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
        
    return model, scaler

model, scaler = load_artifacts()

# --- HELPER FUNCTION FOR FEATURE ENGINEERING ---
def engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the feature engineering from the training pipeline for a single prediction.
    """
    df_featured = df.copy()
    
    # 1. Create new features
    # Add a small epsilon to avoid division by zero
    df_featured['BalanceToSalaryRatio'] = df_featured['Balance'] / (df_featured['EstimatedSalary'] + 1)
    
    # AgeGroup: Use pd.cut and then convert to codes
    age_bins = [0, 30, 45, 60, 100]
    age_labels = ['Young', 'Middle', 'Senior', 'Elder']
    df_featured['AgeGroup'] = pd.cut(df_featured['Age'], bins=age_bins, labels=age_labels)
    # Map labels to the codes used during training
    age_mapping = {'Young': 0, 'Middle': 1, 'Senior': 2, 'Elder': 3}
    df_featured['AgeGroup'] = df_featured['AgeGroup'].map(age_mapping)

    # 2. One-Hot Encode categorical variables (manually for single input)
    # The training script used drop_first=True, so we replicate that behavior.
    # Geography: France is the dropped category
    df_featured['Geography_Germany'] = 1 if df['Geography'].iloc[0] == 'Germany' else 0
    df_featured['Geography_Spain'] = 1 if df['Geography'].iloc[0] == 'Spain' else 0
    
    # Gender: Female is the dropped category
    df_featured['Gender_Male'] = 1 if df['Gender'].iloc[0] == 'Male' else 0

    # 3. Drop original categorical columns
    df_featured = df_featured.drop(['Geography', 'Gender'], axis=1)
    
    # 4. Ensure all columns from training are present and in the correct order
    # This is a crucial step to avoid errors. The list of columns must match the one
    # used to train the scaler and the model.
    expected_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
        'BalanceToSalaryRatio', 'AgeGroup', 'Geography_Germany', 
        'Geography_Spain', 'Gender_Male'
    ]
    
    # Reindex the DataFrame to match the training columns
    df_reindexed = df_featured.reindex(columns=expected_columns, fill_value=0)
    
    return df_reindexed

# --- UI LAYOUT ---

# Header
st.title("🏦 Customer Churn Prediction")
st.markdown("Enter customer details on the left panel to predict the likelihood of churn.")
st.markdown("---")


# Sidebar for user inputs
st.sidebar.header("Customer Details")

def user_input_features():
    """
    Creates sidebar widgets to collect user input.
    """
    geography = st.sidebar.selectbox("Geography", ('France', 'Germany', 'Spain'))
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    
    age = st.sidebar.slider("Age", 18, 100, 35)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
    balance = st.sidebar.number_input("Balance", min_value=0.0, max_value=250000.0, value=75000.0, step=1000.0)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)
    
    has_cr_card = st.sidebar.radio("Has Credit Card?", ('Yes', 'No'), format_func=lambda x: x)
    is_active_member = st.sidebar.radio("Is Active Member?", ('Yes', 'No'), format_func=lambda x: x)

    # Convert inputs to a dictionary
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': estimated_salary
    }
    
    # Create DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# --- MAIN PANEL ---
input_df = user_input_features()

# Display user inputs
st.header("👤 Customer Profile")
st.table(input_df.T.rename(columns={0: 'Values'}))

# Prediction button
if st.button("Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer data..."):
        # 1. Feature Engineering
        engineered_df = engineer_features_for_inference(input_df)
        
        # 2. Scaling
        scaled_features = scaler.transform(engineered_df)
        
        # 3. Prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.header("📈 Prediction Result")
        
        churn_probability = prediction_proba[0][1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("Prediction: **Customer will Churn** 😟")
            else:
                st.success("Prediction: **Customer will Not Churn** 😊")
        
        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability:.2%}",
                delta=f"{churn_probability - 0.5:.2%}",
                delta_color="inverse"
            )
        
        # Display a progress bar for visual representation
        st.progress(churn_probability, text=f"Confidence: {churn_probability:.0%}")
        
        st.info(f"**Explanation:** The model predicts a **{churn_probability:.2%} probability** that this customer will churn. Values closer to 100% indicate a higher risk of churn.")