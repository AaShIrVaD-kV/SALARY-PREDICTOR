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
            
            # Education
            educations = sorted(df_clean['education'].unique())
            education = st.selectbox("Education", educations)

        with col2:
            occupations = sorted(df_clean['occupation'].unique())
            occupation = st.selectbox("Occupation", occupations)
            
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

        # Prepare Input Data
        # Using only 5 selected features: age, education, occupation, workclass, hours-per-week
        input_data = pd.DataFrame({
            'age': [age],
            'education': [education],
            'occupation': [occupation],
            'workclass': [workclass],
            'hours-per-week': [hours_per_week]
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

        # --- PART 1: Handle Target Imbalance ---
        st.markdown("### Part 1: Target Imbalance Handling")

        df_eda = df_clean.copy()
        df_eda["salary_bin"] = (
            df_eda["salary"]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
            .map({"<=50K": 0, ">50K": 1})
        )
        if df_eda["salary_bin"].isna().any():
            st.warning("Some salary values could not be mapped to 0/1. Please check salary labels.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Salary Distribution (Encoded)")
            fig, ax = plt.subplots()
            sns.countplot(x="salary_bin", data=df_eda, ax=ax)
            ax.set_xlabel("Salary (0 = <=50K, 1 = >50K)")
            st.pyplot(fig)

        with col2:
            st.write("#### Percentage Distribution (Encoded)")
            perc = df_eda["salary_bin"].value_counts(normalize=True).sort_index() * 100
            st.write(perc.rename(index={0: "<=50K", 1: ">50K"}).round(2).astype(str) + " %")

        # Balance using SMOTE (preferred) if available
        X = df_eda.drop(columns=["salary", "salary_bin"])
        y = df_eda["salary_bin"]
        X_encoded = pd.get_dummies(X, drop_first=False)

        smote_available = False
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
            smote_available = True
        except Exception:
            smote_available = False

        if smote_available:
            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(X_encoded, y)
            st.success("SMOTE applied to balance classes.")
        else:
            X_bal, y_bal = X_encoded, y
            st.info("SMOTE not available. Install `imbalanced-learn` to enable SMOTE.")

        st.write("#### Class Distribution After Balancing")
        fig_bal, ax_bal = plt.subplots()
        sns.countplot(x=y_bal, ax=ax_bal)
        ax_bal.set_xlabel("Salary (0 = <=50K, 1 = >50K)")
        st.pyplot(fig_bal)
        st.write((pd.Series(y_bal).value_counts(normalize=True).sort_index() * 100).round(2).astype(str) + " %")

        st.markdown("---")
        st.markdown("### Part 2: Feature vs Target Subplots")

        feature_cols = [c for c in df_eda.columns if c not in ["salary", "salary_bin"]]
        n_features = len(feature_cols)
        cols = 3
        rows = int(np.ceil(n_features / cols))

        fig_grid, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(feature_cols):
            ax = axes[i]
            if pd.api.types.is_numeric_dtype(df_eda[col]):
                sns.boxplot(x="salary", y=col, data=df_eda, ax=ax)
            else:
                order = df_eda[col].value_counts().index
                sns.countplot(x=col, hue="salary", data=df_eda, order=order, ax=ax)
                ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"Salary vs {col}")
        for j in range(i + 1, len(axes)):
            fig_grid.delaxes(axes[j])
        st.pyplot(fig_grid)

        st.markdown("---")
        st.markdown("### Part 3: Correlation Analysis")

        df_encoded = pd.get_dummies(df_eda.drop(columns=["salary"]), drop_first=False)
        corr_matrix = df_encoded.corr()
        corr_with_target = corr_matrix["salary_bin"].sort_values(ascending=False)

        st.write("#### Correlation with Salary (Encoded)")
        st.write(corr_with_target)

        threshold = 0.05
        low_corr = corr_with_target.drop("salary_bin")
        low_corr = low_corr[low_corr.abs() < threshold].index.tolist()

        st.write("Dropping low correlated features:")
        st.write(low_corr)

        df_selected = df_encoded.drop(columns=low_corr)

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
