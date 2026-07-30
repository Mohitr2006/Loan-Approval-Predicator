import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
    #for  numerical pipeline
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    #for categorical columns
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    #Construct Full Pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])
    
    return full_pipeline

def preprocess(df):
    
    df.columns = df.columns.str.strip()

    df["loan_status"] = df["loan_status"].str.strip()

    df["loan_status"] = df["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['loan_status']):
        x_train = df.loc[train_index]
        x_test = df.loc[test_index]

    y_train = x_train['loan_status']
    y_test = x_test['loan_status']

    drop_attribs = ['loan_id', 'loan_status']

    x_train = x_train.drop(drop_attribs, axis=1)
    x_test = x_test.drop(drop_attribs, axis=1)

    return x_train, x_test, y_train, y_test

def inference(MODEL_FILE, PIPELINE_FILE):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    df = pd.read_csv("loan_approval_dataset.csv")

    x_train, x_test, y_train, y_test = preprocess(df)

    results = x_test.copy()
    results['loan_status_actual'] = y_test

    transformed_input = pipeline.transform(x_test)
    predictions = model.predict(transformed_input)
    results['loan_status_predicted'] = predictions
    
    results.to_csv("output.csv", index=False)
    print("Inference Completed")

if not (
    os.path.exists(MODEL_FILE)
    and os.path.exists(PIPELINE_FILE)
):

    df = pd.read_csv("loan_approval_dataset.csv")

    x_train, x_test, y_train, y_test = preprocess(df)

    df = x_train.copy()

    num_attribs = df.select_dtypes(include=np.number).columns.tolist()
    cat_attribs = df.select_dtypes(include="object").columns.tolist()

    pipeline = build_pipeline(num_attribs, cat_attribs)

    loan_prepared = pipeline.fit_transform(df)

    model = RandomForestClassifier(random_state=42)
    model.fit(loan_prepared, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained")
    inference(MODEL_FILE, PIPELINE_FILE)

else:
    inference(MODEL_FILE, PIPELINE_FILE)
