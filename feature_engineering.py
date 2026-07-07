import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def total_service (X):
    service_cols = X.loc[:,"PhoneService":"StreamingMovies"].columns
    X["TotalServices"] = X[service_cols].apply(lambda x: len(service_cols) - sum(v == "No" for v in x), axis=1) 
    return X
# Counts the number of services taken by subtracting "No" responses from total number of service columns

def gender_binary (X):
    X = X.copy()
    X["gender"] = X["gender"].map({"Male":1, "Female":0})
    return X

def binary_transform (X):
    X = X.copy()
    binary_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
               "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    for col in binary_cols:
        X[col] = X[col].map({"Yes":1, "No":0})
    return X

total_service_transformer = FunctionTransformer(total_service) # Creates a new column with the number of total service taken
gender_binary_transformer = FunctionTransformer(gender_binary) # Transforms only "Gender" column to binary
yes_no_transformer = FunctionTransformer(binary_transform) # Transforms all "Yes-No" values of according columns to binary