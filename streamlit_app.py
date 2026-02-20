"""
Streamlit Entry Point for Salary Prediction System
This file is at the root level for Streamlit Cloud deployment.
It imports and runs the main app from the CODE folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION - Works for both local and Streamlit Cloud deployment
# ============================================================================

def get_resource_path(folder, filename):
    """
    Get the absolute path to a resource file.
    Works in both local development and Streamlit Cloud environments.
    """
    # Try multiple path strategies
    possible_paths = [
        # Strategy 1: Root level (Streamlit Cloud)
        Path(folder) / filename,
        # Strategy 2: From current working directory
        Path.cwd() / folder / filename,
    ]
    
    # Return the first path that exists
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # If no path exists, return the most likely one (root level for Cloud)
    return str(Path(folder) / filename)


MODEL_PATH = get_resource_path('MODELS', 'best_model.pkl')
ENCODER_PATH = get_resource_path('MODELS', 'label_encoder_target.pkl')
DATA_PATH = get_resource_path('DATA', 'salary.csv')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Salary Prediction System", layout="wide")

# ============================================================================
# CACHE AND LOAD FUNCTIONS
# ============================================================================
@st.cache_data
def load_data():
    """Load dataset with proper handling of missing values."""
    try:
        df = pd.read_csv(DATA_PATH, skipinitialspace=True, na_values='?')
        return df
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found at: {DATA_PATH}")
        st.error(f"Error: {e}")
        return None

@st.cache_resource
def load_model():
    """Load trained model and target encoder."""
    try:
        model = joblib.load(MODEL_PATH)
        le_target = joblib.load(ENCODER_PATH)
        return model, le_target
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found!")
        st.error(f"Looking for: {MODEL_PATH}")
        st.error(f"Error: {e}")
        return None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("💰 Salary Prediction System")
    st.markdown("""
    Welcome to the Salary Prediction System. This application predicts whether a candidate's salary will be 
    **<=50K** or **>50K** based on their professional and demographic details.
    """)

    # Load data and model
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the deployment configuration.")
        return
    
    model, le_target = load_model()
    if model is None or le_target is None:
        st.error("Failed to load model. Please check the deployment configuration.")
        return
    
    # Clean df for EDA (drop missing) same as training
    df_clean = df.dropna()

    # Sidebar Navigation
    menu = ["Prediction", "Exploratory Data Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Predict Salary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=100, value=35)
            workclass_options = sorted(df_clean['workclass'].unique())
            workclass = st.selectbox("Workclass", workclass_options)
            education_options = sorted(df_clean['education'].unique())
            education = st.selectbox("Education", education_options)
        
        with col2:
            occupation_options = sorted(df_clean['occupation'].unique())
            occupation = st.selectbox("Occupation", occupation_options)
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
        
        if st.button("Predict Salary", key="predict_btn", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'education': [education],
                'occupation': [occupation],
                'workclass': [workclass],
                'hours-per-week': [hours_per_week]
            })
            
            # Make prediction
            try:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                
                # Decode prediction
                salary_label = le_target.inverse_transform(prediction)[0]
                confidence = max(prediction_proba[0]) * 100
                
                st.success(f"✅ **Prediction: {salary_label}**")
                st.info(f"📊 **Confidence: {confidence:.2f}%**")
                
                # Show detailed probabilities
                st.subheader("Prediction Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("≤50K Probability", f"{prediction_proba[0][0]*100:.2f}%")
                with col2:
                    st.metric(">50K Probability", f"{prediction_proba[0][1]*100:.2f}%")
                
                # Show input summary
                st.subheader("Input Summary")
                st.json({
                    "Age": age,
                    "Education": education,
                    "Occupation": occupation,
                    "Workclass": workclass,
                    "Hours per Week": hours_per_week
                })
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
    
    elif choice == "Exploratory Data Analysis":
        st.subheader("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_clean))
        with col2:
            st.metric("Features", df_clean.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        
        # Dataset Preview
        st.subheader("Dataset Preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
        # Target Distribution (After Balancing Note)
        st.subheader("Target Variable Distribution")
        salary_counts = df_clean['salary'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            salary_counts.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
            ax.set_title('Salary Distribution (Original Data)')
            ax.set_xlabel('Salary Category')
            ax.set_ylabel('Count')
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            salary_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
            ax.set_title('Salary Distribution (Percentage)')
            ax.set_ylabel('')
            st.pyplot(fig)
        
        st.info("""
        **Note:** The model was trained on SMOTE-balanced data (50:50 ratio) to improve minority class 
        prediction accuracy. The original data shows class imbalance, but the model learns equally from both classes.
        """)
        
        # Feature Analysis
        st.subheader("Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Education Distribution**")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_clean['education'].value_counts().plot(kind='barh', ax=ax, color='#3498db')
            ax.set_xlabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Occupation Distribution**")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_clean['occupation'].value_counts().plot(kind='barh', ax=ax, color='#2ecc71')
            ax.set_xlabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Age Distribution
        st.markdown("**Age Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df_clean['age'], bins=30, color='#9b59b6', edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution')
        st.pyplot(fig)
        
        # Hours per Week Distribution
        st.markdown("**Hours per Week Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df_clean['hours-per-week'], bins=30, color='#f39c12', edgecolor='black')
        ax.set_xlabel('Hours per Week')
        ax.set_ylabel('Frequency')
        ax.set_title('Hours per Week Distribution')
        st.pyplot(fig)
        
        # Workclass Distribution
        st.markdown("**Workclass Distribution**")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_clean['workclass'].value_counts().plot(kind='bar', ax=ax, color='#e67e22')
        ax.set_title('Workclass Distribution')
        ax.set_xlabel('Workclass')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
