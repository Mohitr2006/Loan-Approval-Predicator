import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("loan_approval_dataset.csv")

df.columns = df.columns.str.strip()

df["loan_status"] = df["loan_status"].str.strip()

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['loan_status']):
    x_train = df.loc[train_index]
    x_test = df.loc[test_index]

y_train = x_train['loan_status']
y_test = x_test['loan_status']

drop_attribs = ['loan_id', 'loan_status']

for dataset in (x_train, x_test):
    dataset.drop(drop_attribs, axis=1, inplace=True)

df = x_train.copy()

loan_num = df.select_dtypes(include=(np.number))
num_attribs = loan_num.columns.tolist()

loan_cat = df.copy()
loan_cat.drop(num_attribs, axis=1, inplace=True)
cat_attribs = loan_cat.columns.tolist()

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
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

loan_prepared = full_pipeline.fit_transform(df)

# 7. Train the model

# Linear Regression Model
print("Logistic Regression")
log_reg = LogisticRegression()
scores = cross_val_score(
    log_reg,
    loan_prepared,
    y_train,
    scoring="accuracy",
    cv=10
)
print(pd.Series(scores).describe())

# Decision Tree Model
print("Decision Tree Classifier")
dec_cls = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(
    dec_cls,
    loan_prepared,
    y_train,
    scoring="accuracy",
    cv=10
)
print(pd.Series(scores).describe())

# Random Forest Regressor
print("Random Forest Classifier")
ran_cls = RandomForestClassifier(random_state=42)
scores = cross_val_score(
    ran_cls,
    loan_prepared,
    y_train,
    scoring="accuracy",
    cv=10
)
print(pd.Series(scores).describe())