import warnings
warnings.filterwarnings("ignore")

import sys
import random
import numpy as np
import pandas as pd

from rf import RandomForestClassifier
from rf_preprocess import OneHotEncoder, MinMaxScaler, CarsTransformer

import sklearn.ensemble as ensemble



datasets_path = './dataset_files/'
datasets = ['cancer', 'car']
scalers = {'cancer': [None, MinMaxScaler], 'car': [None, OneHotEncoder, CarsTransformer]}

def get_datasets(data):
    X_train = pd.read_csv(datasets_path+data+'_X_train.csv')
    X_test = pd.read_csv(datasets_path+data+'_X_test.csv')
    y_train = pd.read_csv(datasets_path+data+'_y_train.csv')
    y_test = pd.read_csv(datasets_path+data+'_y_test.csv')
    return(X_train, X_test, y_train, y_test)


def rf_test(dataset, scale=None):
    print()
    print("Dataset: ", dataset)
    reg = RandomForestClassifier()
    X_train, X_test, y_train, y_test = get_datasets(dataset)
    if(scale is not None):
        scaler = scale()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    reg.fit(X_train, y_train)
    print("Train Accuracy: ", reg.score(X_train, y_train))
    accuracy = reg.score(X_test, y_test)
    print("Test Accuracy: ", accuracy)
    reg.confusion(X_test, y_test)
    reg.metrics(X_test, y_test)
    return(accuracy)

def scikit_test(dataset, scale=None):
    reg = ensemble.RandomForestClassifier()
    X_train, X_test, y_train, y_test = get_datasets(dataset)
    if(scale is not None):
        scaler = scale()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    reg.fit(X_train, y_train)
    accuracy = reg.score(X_test, y_test)
    return(accuracy)

def consolidate_results():
    res = pd.DataFrame(columns=['Dataset', 'Scale', 'Accuracy', 'Scikit Score'])
    for dataset in datasets:
        scaler = scalers[dataset][1]
        rf_acc = rf_test(dataset, scale=scaler)
        try:
            sk_acc = scikit_test(dataset, scale=scaler)
        except:
            sk_acc = np.nan
        res = res.append({"Dataset": dataset,"Scale": scaler , "Accuracy": rf_acc,
                    "Scikit Score":sk_acc}, ignore_index=True)
    print(res)

def main():
    # rf_test('cancer')
    # rf_test('cancer', MinMaxScaler)
    # rf_test('car', CarsTransformer)
    # rf_test('car')
    consolidate_results()

main()
