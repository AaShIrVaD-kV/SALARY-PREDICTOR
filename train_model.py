import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- Configuration ---
DATA_PATH = "salary.csv"
MODEL_PATH = "best_model.pkl"
ENCODER_PATH = "label_encoder_target.pkl"

def load_data(path):
    """Load dataset, handline missing values marked as ' ?'."""
    # Data has spaces after commas, so we need skipinitialspace=True.
    # Also ' ?' is used for missing values.
    
    # Read first line to check headers. It seems to have headers.
    df = pd.read_csv(path, skipinitialspace=True, na_values='?')
    return df

def preprocess_data(df):
    """Clean and prepare data for training."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Drop irrelevant columns
    # 'fnlwgt' is sampling weight, not predictive for individuals usually.
    # 'education' is redundant with 'education-num'.
    df = df.drop(['fnlwgt', 'education'], axis=1)
    
    # Separate Features and Target
    X = df.drop('salary', axis=1)
    y = df['salary']
    
    return X, y

def get_preprocessor():
    """Create a ColumnTransformer for preprocessing."""
    
    # Numerical features
    numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    numerical_transformer = StandardScaler()
    
    # Categorical features
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """Train multiple models and select the best one."""
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    results = {}

    print("Training models...\n")
    
    for name, model in models.items():
        # Create pipeline: Preprocess -> Model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 30)
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = clf
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")
    return best_model

def main():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Encode Target (salary) manually or with LabelEncoder to be safe
    # <=50K -> 0, >50K -> 1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the target encoder for inverse transform in App
    joblib.dump(le, ENCODER_PATH)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Get Preprocessor
    preprocessor = get_preprocessor()
    
    # Train and Evaluate
    final_model = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    # Save Model
    print(f"Saving best model to {MODEL_PATH}...")
    joblib.dump(final_model, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
