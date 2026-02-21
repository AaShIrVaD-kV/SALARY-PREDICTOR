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
import os

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'salary.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'MODELS', 'best_model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), '..', 'MODELS', 'label_encoder_target.pkl')

# Selected features (must match what app.py sends to model.predict)
SELECTED_FEATURES = ['age', 'education', 'occupation', 'workclass', 'hours-per-week']
NUMERICAL_FEATURES = ['age', 'hours-per-week']
CATEGORICAL_FEATURES = ['education', 'occupation', 'workclass']

def load_data(path):
    """Load dataset, handling missing values marked as '?'."""
    df = pd.read_csv(path, skipinitialspace=True, na_values='?')
    return df

def preprocess_data(df):
    """Clean and prepare data for training."""
    df = df.dropna()
    selected = SELECTED_FEATURES + ['salary']
    df = df[selected]
    X = df.drop('salary', axis=1)
    y = df['salary']
    return X, y

def get_preprocessor():
    """Create a ColumnTransformer for preprocessing."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
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
        # Pipeline: preprocessor is already fitted, so we just need classifier here
        # We use a new pipeline per model that includes the (already-fitted) preprocessor
        # by wrapping in a fresh pipeline and manually passing pre-processed data.
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
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
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = load_data(DATA_PATH)
    print(f"Dataset loaded: {df.shape}")

    print("Preprocessing data...")
    X, y = preprocess_data(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # --- Encode target ---
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"\nTarget classes: {le_target.classes_}")

    # --- CLASS DISTRIBUTION BEFORE BALANCING ---
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION - BEFORE BALANCING")
    print("=" * 80)

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
    print("⚠️  DATA IS IMBALANCED - Applying SMOTE after one-hot encoding...")

    # --- ENCODE CATEGORICAL FEATURES FIRST (needed for SMOTE) ---
    print("\n" + "=" * 80)
    print("ENCODING CATEGORICAL FEATURES FOR SMOTE")
    print("=" * 80)

    # Fit preprocessor on full data to encode before SMOTE
    temp_preprocessor = get_preprocessor()
    X_encoded = temp_preprocessor.fit_transform(X)
    print(f"Encoded feature matrix shape: {X_encoded.shape}")

    # Get encoded column names for reference
    ohe = temp_preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    print(f"Total encoded features: {len(all_feature_names)}")

    # --- APPLY SMOTE (on encoded data) ---
    print("\n" + "=" * 80)
    print("APPLYING SMOTE BALANCING")
    print("=" * 80)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_encoded, y_encoded)

    print(f"\nOriginal dataset size: {X_encoded.shape[0]}")
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

    # Convert back to DataFrame with feature names
    X_balanced_df = pd.DataFrame(X_balanced, columns=all_feature_names)

    X_train_enc, X_test_enc, y_train, y_test = train_test_split(
        X_balanced_df, y_balanced, test_size=0.2, random_state=42
    )

    # --- Because data is already encoded, we build a pipeline WITHOUT a preprocessor ---
    # and instead save a fresh pipeline WITH preprocessor fitted on original X for inference.
    # Strategy: Train classifier on encoded data, but save a full pipeline (preprocessor + clf)
    # fitted on original (un-encoded) X so app.py can send raw string features.

    print("Training classifiers on SMOTE-balanced encoded data...\n")

    trained_classifiers = {}
    results = {}

    classifier_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42, learning_rate=0.1, max_depth=5)
    }

    for name, clf in classifier_models.items():
        clf.fit(X_train_enc, y_train)
        y_pred = clf.predict(X_test_enc)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_classifiers[name] = clf

        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 30)

    # Pick best classifier
    best_name = max(results, key=results.get)
    best_clf = trained_classifiers[best_name]
    print(f"\nBest Model: {best_name} with Accuracy: {results[best_name]:.4f}")

    # --- BUILD FINAL INFERENCE PIPELINE ---
    # Fit a fresh preprocessor on original X (raw strings), attach best classifier.
    # This pipeline will accept raw feature DataFrames from app.py.
    print("\nBuilding final inference pipeline with preprocessor on original data...")
    inference_preprocessor = get_preprocessor()
    inference_preprocessor.fit(X)

    final_pipeline = Pipeline(steps=[
        ('preprocessor', inference_preprocessor),
        ('classifier', best_clf)
    ])

    # Quick sanity check
    sample = X.iloc[:5]
    preds = final_pipeline.predict(sample)
    print(f"Sanity check predictions on 5 samples: {preds}")
    print(f"Expected labels: {le_target.inverse_transform(preds)}")

    # --- SAVE MODEL ---
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(final_pipeline, MODEL_PATH)

    print(f"Saving target encoder to {ENCODER_PATH}...")
    joblib.dump(le_target, ENCODER_PATH)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\nSUMMARY:")
    print(f"  ✓ Original Imbalance: 1:{imbalance_ratio:.2f}")
    print(f"  ✓ After SMOTE Balancing: 1:1 (Perfect)")
    print(f"  ✓ Features Used: {', '.join(SELECTED_FEATURES)}")
    print(f"  ✓ Best Model: {best_name} (Accuracy: {results[best_name]:.4f})")
    print(f"  ✓ Model Saved: {MODEL_PATH}")
    print(f"  ✓ Encoder Saved: {ENCODER_PATH}")

if __name__ == "__main__":
    main()
