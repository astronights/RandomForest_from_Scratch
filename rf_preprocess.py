from abc import ABC
import pandas as pd

class MinMaxScaler(ABC):

    def __init__(self):
        self.mm = {}

    def fit_transform(self, data):
        for col in data.columns:
            self.mm[col] = (min(data[col]), max(data[col]))
            data[col] = (data[col] - self.mm[col][0])/(self.mm[col][1] - self.mm[col][0])
        return(data)

    def transform(self, data):
        for col in data.columns:
            data[col] = (data[col] - self.mm[col][0])/(self.mm[col][1] - self.mm[col][0])
        return(data)

class OneHotEncoder(ABC):

    def __init__(self):
        self.cols = []

    def fit_transform(self, data):
        ret_data = pd.get_dummies(data)
        self.cols = ret_data.columns
        return(ret_data)

    def transform(self, data):
        ret_data = pd.get_dummies(data)
        cols = set(ret_data.columns) - set(self.cols)
        for col in cols:
            ret_data[col] = 0
        return(ret_data)

class CarsTransformer(ABC):

    def __init__(self):
        self.x_vals = {'low': 0, 'med': 1, 'high': 2,
                     'vhigh': 3, 'small': 0, 'big': 2,
                     '2': 1, '3' : 2, '4' : 3,
                     '5more' : 4, 'more': 5}

    def fit_transform(self, data):
        for col in data.columns:
            data[col] = data[col].map(self.x_vals)
        return(data)

    def transform(self, data):
        for col in data.columns:
            data[col] = data[col].map(self.x_vals)
        return(data)
