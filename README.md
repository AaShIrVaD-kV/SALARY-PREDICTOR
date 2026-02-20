# Salary Prediction System

## Project Overview
This is a Machine Learning-based system designed to predict whether a candidate's salary will exceed $50K/year based on census data. The project involves data preprocessing, exploratory data analysis (EDA), model training with multiple algorithms, and a user-friendly Streamlit web application for real-time predictions.

## Dataset
The system uses the **Adult Census Income Dataset**, containing attributes like:
- Age, Workclass, Education
- Marital Status, Occupation, Relationship
- Race, Sex
- Capital Gain, Capital Loss
- Hours per Week, Native Country

## System Modules
1. **Data Preprocessing**: Handles missing values (`?`), encodes categorical variables (One-Hot Encoding), and scales numerical features.
2. **Model Training**: Implements and compares:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
3. **Web Application**: Interactive interface built with Streamlit.

## Installation
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
To retrain the models and generate the best model file:
```bash
cd CODE
python train_model.py
```
This script will:
- Load `salary.csv` from `DATA/` folder.
- Preprocess the data (feature selection, balancing with SMOTE).
- Train Logistic Regression, Random Forest, and Gradient Boosting models.
- Print comparative accuracy metrics.
- Save the best model to `MODELS/best_model.pkl` and the target encoder to `MODELS/label_encoder_target.pkl`.

### 2. Running the Web App
To launch the prediction interface:
```bash
cd CODE
streamlit run app.py
```
- Navigate to **Exploratory Data Analysis** to view data insights.
- Go to **Prediction** to input candidate details and get a salary forecast.

## Model Performance
The system evaluates models based on balanced accuracy after applying SMOTE balancing:
- **Gradient Boosting Classifier**: 81.49% Accuracy (BEST - Excellent minority class recall at 85%)
- **Random Forest Classifier**: 79.21% Accuracy (Good balance with 84% minority recall)
- **Logistic Regression**: 74.26% Accuracy (Balanced performance)

**Key Improvement Features:**
- ✅ SMOTE data balancing (perfect 1:1 class ratio)
- ✅ Optimized 5-feature selection (age, education, occupation, workclass, hours-per-week)
- ✅ Removed fairness concerns (race, gender, sensitive attributes)
- ✅ Improved minority class recall: 61% → 85% (+24 points)

## Project Structure
```
SALARY PREDICTOR/
├── CODE/                              # Python source code
│   ├── app.py                        # Streamlit web application
│   └── train_model.py                # Model training pipeline with SMOTE
├── DATA/                             # Dataset files
│   └── salary.csv                    # Adult Census Income Dataset
├── MODELS/                           # Trained model artifacts
│   ├── best_model.pkl                # Best trained Gradient Boosting model
│   └── label_encoder_target.pkl      # Target variable (salary) encoder
├── VISUALIZATIONS/                   # Generated charts and plots
│   └── class_balance_comparison.png  # Before/after SMOTE comparison
├── DOCS/                             # Documentation files
│   ├── ENHANCEMENT_REPORT.md         # Detailed improvement report with all metrics
│   ├── CLEANUP_SUMMARY.md            # Files removed and why
│   └── PROJECT_DOCUMENTATION.md      # Complete project overview
├── requirements.txt                   # Python package dependencies
├── runtime.txt                        # Python version specification
└── README.md                          # This file
```

---
**Developed for Final Year Project Demonstration**
