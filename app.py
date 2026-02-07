import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constants
MODEL_PATH = "best_model.pkl"
ENCODER_PATH = "label_encoder_target.pkl"
DATA_PATH = "salary.csv"

# Page Config
st.set_page_config(page_title="Salary Prediction System", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True, na_values='?')
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le_target = joblib.load(ENCODER_PATH)
    return model, le_target

def main():
    st.title("💰 Salary Prediction System")
    st.markdown("""
    Welcome to the Salary Prediction System. This application predicts whether a candidate's salary will be 
    **<=50K** or **>50K** based on their professional and demographic details.
    """)

    # Load data for options and EDA
    df = load_data()
    model, le_target = load_model()
    
    # Clean df for EDA (drop missing) same as training
    df_clean = df.dropna()

    # Sidebar Navigation
    menu = ["Prediction", "Exploratory Data Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Predict Salary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=30)
            
            # Workclass
            workclasses = sorted(df_clean['workclass'].unique())
            workclass = st.selectbox("Workclass", workclasses)
            
            # Education - Map name to number
            # Create a mapping dictionary from the clean dataframe
            # Sort by education-num to make the list logical
            edu_df = df_clean[['education', 'education-num']].drop_duplicates().sort_values('education-num')
            education_options = edu_df['education'].tolist()
            education_mapping = dict(zip(edu_df['education'], edu_df['education-num']))
            
            education_level = st.selectbox("Education Level", education_options, index=9) # Default to some-college or bachelors
            education_num = education_mapping[education_level]

            marital_statuses = sorted(df_clean['marital-status'].unique())
            marital_status = st.selectbox("Marital Status", marital_statuses)
            
            occupations = sorted(df_clean['occupation'].unique())
            occupation = st.selectbox("Occupation", occupations)
            
            relationships = sorted(df_clean['relationship'].unique())
            relationship = st.selectbox("Relationship", relationships)

        with col2:
            races = sorted(df_clean['race'].unique())
            race = st.selectbox("Race", races)
            
            sexes = sorted(df_clean['sex'].unique())
            sex = st.selectbox("Sex", sexes)
            
            capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
            capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
            
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
            
            native_countries = sorted(df_clean['native-country'].unique())
            # Default to United-States if available
            default_country_idx = native_countries.index('United-States') if 'United-States' in native_countries else 0
            native_country = st.selectbox("Native Country", native_countries, index=default_country_idx)

        # Prepare Input Data
        # Must match columns: ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'education-num': [education_num], # IMPORTANT: Use the number
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'sex': [sex],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })
        
        if st.button("Predict Salary"):
            st.write("Processing input...")
            try:
                prediction_idx = model.predict(input_data)[0]
                prediction_label = le_target.inverse_transform([prediction_idx])[0]
                
                # Probability
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_data)[0]
                    confidence = proba[prediction_idx] * 100
                else:
                    confidence = 0

                st.write("---")
                if prediction_label == ">50K":
                    st.success(f"### Result: High Income (>50K)")
                else:
                    st.warning(f"### Result: Low/Medium Income (<=50K)")
                
                if confidence > 0:
                     st.info(f"Confidence: {confidence:.2f}%")
                     
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    elif choice == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.head(20))
            st.write(f"Shape of Dataset: {df.shape}")

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Salary Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='salary', data=df_clean, ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("#### Imbalance Check")
            st.write(df_clean['salary'].value_counts(normalize=True))

        st.write("#### Income by Education Level")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(y='education', hue='salary', data=df_clean, order=df_clean['education'].value_counts().index, ax=ax2)
        st.pyplot(fig2)
        
        st.write("#### Income by Workclass")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.countplot(y='workclass', hue='salary', data=df_clean, order=df_clean['workclass'].value_counts().index, ax=ax3)
        st.pyplot(fig3)

        st.markdown("---")
        st.write("### Advanced Visualizations")
        
        # Correlation Heatmap
        st.write("#### Correlation Heatmap")
        st.info("Correlation between numerical features.")
        numeric_df = df_clean.select_dtypes(include=[np.number])
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        # Outliers
        st.write("#### Outlier Detection (Boxplots)")
        outlier_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
        outlier_col_select = st.selectbox("Select Feature for Outlier Visualization", outlier_cols)
        
        fig_out, ax_out = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df_clean[outlier_col_select], color='orange', ax=ax_out)
        st.pyplot(fig_out)

        # Relationship with Target
        st.write("#### Numerical Features vs Salary")
        num_vs_salary_col = st.selectbox("Select Feature to Compare with Salary", outlier_cols, index=0)
        fig_rel, ax_rel = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='salary', y=num_vs_salary_col, data=df_clean, palette="Set2", ax=ax_rel)
        st.pyplot(fig_rel)


if __name__ == '__main__':
    main()
