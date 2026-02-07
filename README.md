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
python train_model.py
```
This script will:
- Load `salary.csv`.
- Preprocess the data.
- Train Logistic Regression, Decision Tree, and Random Forest models.
- Print comparative accuracy metrics.
- Save the best model to `best_model.pkl` and the target encoder to `label_encoder_target.pkl`.

### 2. Running the Web App
To launch the prediction interface:
```bash
streamlit run app.py
```
- Navigate to **Exploratory Data Analysis** to view data insights.
- Go to **Prediction** to input candidate details and get a salary forecast.

## Model Performance
The system evaluates models based on Accuracy. Typical results:
- **Logistic Regression**: ~85.3% Accuracy (Selected as Best Model)
- **Random Forest Classifier**: ~84.3% Accuracy
- **Decision Tree Classifier**: ~81.4% Accuracy

## Project Structure
- `app.py`: Streamlit dashboard and prediction engine.
- `train_model.py`: Training pipeline and evaluation script.
- `salary.csv`: Historical dataset.
- `requirements.txt`: Python package dependencies.
- `best_model.pkl`: Serialized trained model.
- `label_encoder_target.pkl`: Encoder for target labels.

---
**Developed for Final Year Project Demonstration**
