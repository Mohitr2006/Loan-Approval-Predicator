# Loan Approval Prediction using Machine Learning

## Overview

This project predicts whether a loan application will be **Approved** or **Rejected** using a **Random Forest Classifier**. It demonstrates a complete end-to-end machine learning workflow, from data preprocessing to model training, saving, loading, and inference.

## Features

- Data preprocessing and cleaning
- Stratified Train-Test Split
- Feature scaling using `StandardScaler`
- One-Hot Encoding for categorical features
- Scikit-learn `Pipeline` and `ColumnTransformer`
- Random Forest Classifier
- Model serialization using `Joblib`
- Prediction on unseen data
- Export predictions to a CSV file

---

## Dataset

The dataset contains information about loan applicants, including:

- Number of Dependents
- Education
- Self Employment Status
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Asset Value
- Commercial Asset Value
- Luxury Asset Value
- Bank Asset Value

**Target Variable**

- `loan_status`
  - Approved
  - Rejected

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Project Structure

```text
Loan-Approval-Prediction/
│
├── train_model.py
├── loan_approval_dataset.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── model.pkl          (Generated after training)
├── pipeline.pkl       (Generated after training)
└── output.csv         (Generated after inference)
```

---

## Machine Learning Workflow

1. Load the dataset
2. Clean and preprocess the data
3. Perform Stratified Train-Test Split
4. Build preprocessing pipeline
5. Train Random Forest Classifier
6. Save the trained model and preprocessing pipeline
7. Load the saved model
8. Perform inference on the test dataset
9. Save predictions to `output.csv`

---

## Model Performance

**Algorithm**

- Random Forest Classifier

**Test Accuracy**

- **98.13%**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Mohitr2006/Loan-Approval-Predicator.git
```

Move into the project directory:

```bash
cd Loan-Approval-Predicator
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project:

```bash
python train_model.py
```

The script will:

- Train the model if it doesn't already exist.
- Save the trained model (`model.pkl`).
- Save the preprocessing pipeline (`pipeline.pkl`).
- Perform inference on the test data.
- Generate `output.csv` containing actual and predicted loan statuses.

---

## Output

The generated `output.csv` contains:

- Applicant features
- Actual loan status
- Predicted loan status

This makes it easy to compare the model's predictions with the actual labels.

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Model comparison with XGBoost and LightGBM
- Web interface using Flask or Streamlit
- REST API for predictions

---

## Author

**Mohit Raj**

Aspiring Data Scientist passionate about Machine Learning, Python, and AI.
