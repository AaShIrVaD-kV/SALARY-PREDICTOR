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

### 2. Running the Web App (Local)
To launch the prediction interface locally:
```bash
streamlit run streamlit_app.py
```
This will:
- Open the Streamlit web interface in your default browser (usually http://localhost:8501)
- Allow you to navigate between **Prediction** and **Exploratory Data Analysis** tabs
- Make real-time predictions based on user input

### 3. Deployment (Streamlit Cloud)
To deploy on Streamlit Cloud:
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Deploy!

**Note:** The app is configured to work with the folder structure:
- `streamlit_app.py` at the root level (entry point)
- `CODE/app.py` and `CODE/train_model.py` (source code)
- `DATA/salary.csv` (dataset)
- `MODELS/best_model.pkl` and encoder files

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
├── .streamlit/                       # Streamlit configuration
│   └── config.toml                  # Theme and server settings
├── CODE/                            # Python source code
│   ├── app.py                       # Original Streamlit app (can be used locally)
│   └── train_model.py               # Model training pipeline with SMOTE
├── DATA/                            # Dataset files
│   └── salary.csv                   # Adult Census Income Dataset
├── MODELS/                          # Trained model artifacts
│   ├── best_model.pkl               # Best trained Gradient Boosting model
│   └── label_encoder_target.pkl     # Target variable (salary) encoder
├── VISUALIZATIONS/                  # Generated charts and plots
│   └── class_balance_comparison.png # Before/after SMOTE comparison
├── DOCS/                            # Documentation files
│   ├── ENHANCEMENT_REPORT.md        # Detailed improvement report with all metrics
│   ├── CLEANUP_SUMMARY.md           # Files removed and why
│   └── PROJECT_DOCUMENTATION.md     # Complete project overview
├── streamlit_app.py                 # ⭐ ROOT ENTRY POINT for Streamlit (Cloud & Local)
├── requirements.txt                 # Python package dependencies
├── runtime.txt                      # Python version specification
└── README.md                        # This file
```

**Key Points:**
- `streamlit_app.py` is the main entry point for **both local and cloud deployments**
- This file handles all path resolution for different environments
- `CODE/app.py` contains the original implementation (can still be used locally)
- All resource paths are automatically detected for maximum compatibility

---
**Developed for Final Year Project Demonstration**
