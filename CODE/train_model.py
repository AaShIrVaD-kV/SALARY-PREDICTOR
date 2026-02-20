import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / 'DATA' / 'salary.csv'
MODEL_PATH = PROJECT_ROOT / 'MODELS' / 'best_model.pkl'
ENCODER_PATH = PROJECT_ROOT / 'MODELS' / 'label_encoder_target.pkl'

def load_data(path):
    """Load dataset, handline missing values marked as ' ?'."""
    # Data has spaces after commas, so we need skipinitialspace=True.
    # Also ' ?' is used for missing values.
    
    # Read first line to check headers. It seems to have headers.
    df = pd.read_csv(str(path), skipinitialspace=True, na_values='?')
    return df

def preprocess_data(df):
    """Clean and prepare data for training."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Keep only selected high-impact features
    # Features: age, education, occupation, workclass, hours-per-week
    selected_features = ['age', 'education', 'occupation', 'workclass', 'hours-per-week', 'salary']
    df = df[selected_features]
    
    # Separate Features and Target
    X = df.drop('salary', axis=1)
    y = df['salary']
    
    return X, y

def get_preprocessor():
    """Create a ColumnTransformer for preprocessing."""
    
    # Numerical features (selected high-impact features only)
    numerical_features = ['age', 'hours-per-week']
    numerical_transformer = StandardScaler()
    
    # Categorical features (selected high-impact features only)
    categorical_features = ['education', 'occupation', 'workclass']
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
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42, learning_rate=0.1, max_depth=5)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    results = {}

    print("Training models with balanced data (SMOTE)...\n")
    
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
    print("=" * 80)
    print("SALARY PREDICTION MODEL - WITH SMOTE DATA BALANCING")
    print("=" * 80)
    
    print("\nLoading data...")
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = load_data(str(DATA_PATH))
    print(f"Dataset loaded: {df.shape}")
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # --- CLASS DISTRIBUTION BEFORE BALANCING ---
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION - BEFORE BALANCING")
    print("=" * 80)
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    class_counts = pd.Series(y_encoded).value_counts().sort_index()
    print("\nClass Counts:")
    print(class_counts)
    
    class_pct = pd.Series(y_encoded).value_counts(normalize=True).sort_index() * 100
    print("\nClass Distribution (%):")
    for idx, pct in class_pct.items():
        label = "<=50K" if idx == 0 else ">50K"
        print(f"  {label}: {pct:.2f}%")
    
    imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[1]
    print(f"\nImbalance Ratio: 1:{imbalance_ratio:.2f}")
    print("⚠️  DATA IS IMBALANCED - Applying SMOTE...")
    
    # --- APPLY SMOTE ---
    print("\n" + "=" * 80)
    print("APPLYING SMOTE BALANCING")
    print("=" * 80)
    
    # Apply SMOTE before preprocessing (on original features)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y_encoded)
    
    print(f"\nOriginal dataset size: {X.shape[0]}")
    print(f"Balanced dataset size: {X_balanced.shape[0]}")
    
    class_counts_after = pd.Series(y_balanced).value_counts().sort_index()
    class_pct_after = pd.Series(y_balanced).value_counts(normalize=True).sort_index() * 100
    
    print("\nClass Distribution After SMOTE:")
    for idx, count in class_counts_after.items():
        label = "<=50K" if idx == 0 else ">50K"
        pct = class_pct_after[idx]
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print("\n✅ Data is now perfectly balanced (1:1)")
    
    # --- SPLIT DATA ---
    print("\n" + "=" * 80)
    print("SPLITTING DATA AND TRAINING MODELS")
    print("=" * 80 + "\n")
    
    # Convert back to DataFrame for pipeline (needs column names)
    X_balanced_df = pd.DataFrame(X_balanced)
    X_balanced_df.columns = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced_df, y_balanced, test_size=0.2, random_state=42
    )
    
    # Get fresh preprocessor (not fitted yet)
    preprocessor = get_preprocessor()
    
    # Train and Evaluate
    final_model = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    # --- SAVE MODEL ---
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(final_model, str(MODEL_PATH))
    
    print(f"Saving target encoder to {ENCODER_PATH}...")
    joblib.dump(le_target, str(ENCODER_PATH))
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\nSUMMARY:")
    print(f"  ✓ Original Imbalance: 1:{imbalance_ratio:.2f}")
    print(f"  ✓ After SMOTE Balancing: 1:1 (Perfect)")
    print(f"  ✓ Features Used: age, education, occupation, workclass, hours-per-week")
    print(f"  ✓ Model Saved: {str(MODEL_PATH)}")
    print(f"  ✓ Encoder Saved: {str(ENCODER_PATH)}")

if __name__ == "__main__":
    main()
