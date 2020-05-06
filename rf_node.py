import random
import operator
import numpy as np
import pandas as pd

from scipy import stats

class Node:
    def __init__(self, X, y, feats, metric='gini'):
        self.leaf = True
        self.X = X
        self.y = y
        self.features = feats
        self.criteria = None
        self.value = None
        self.max_val = (stats.mode(y)[0][0])
        self.is_cat = False
        self.values = self.get_values()
        self.samples = y.shape[0]
        self.metric = metric
        self.impurity = self.calc_criterion()
        self.left = None
        self.right = None
        self.tried = False

    def _gini(self, p):
        return(p*(1-p) + (1-p)*(1-(1-p)))

    def _entropy(self, p):
        return(-p*np.log2(p) - (1-p)*np.log2(1-p))

    def _error(self, p):
        return(1-np.max([p, 1-p]))

    def info_gain(self, left, right):
        return(self.impurity - \
               (left.samples*left.impurity/self.samples) - \
               (right.samples*right.impurity/self.samples))

    def print_node(self):
        if(self.leaf):
            print('Leaf node,', end=' ')
        else:
            print(self.criteria, end=' ')
            if(self.is_cat):
                print("== " + str(self.value), end=' ')
            else:
                print("<= " + str(self.value), end=' ')
        print(", size: " + str(self.samples) + ', imp: '+ str(self.calc_criterion()) + ', vals: ', self.values)

    def calc_criterion(self):
        method = getattr(self, '_'+self.metric)
        val = (sum([method(self.values[x]/self.samples) for x in self.values]))
        return(val)
        # return(sum([method(x/self.samples) for x in self.values]))

    def get_values(self):
        vals, counts = np.unique(self.y, return_counts=True)
        return(dict(zip(vals, counts)))
        # return(self.y.iloc[:,0].value_counts(sort=False).tolist())

    def best_class(self, root):
        temp = {}
        for key in self.values:
            temp[key] = self.values[key]/root.values[key]
        return(max(temp.items(), key=operator.itemgetter(1))[0])

    def split_node(self, min_sam, num_splits):
        max_info_gain = 0
        self.tried = True
        is_cat = (self.X.dtype == 'object')
        values = []
        if(is_cat):
            for i in range(self.X.shape[1]):
                temp_arr = (np.stack((self.X[:,i], self.y), axis=-1))
                vals = list(set(self.X[:,i]))
                if(len(vals) == 1):
                    continue
                elif(len(vals) == 2):
                    values.append((i, vals[1]))
                else:
                    for x in vals:
                        values.append((i, x))
            # print(values)
            # print()
            # print("Calculating splits...")
            for i, val in values:
                X_left = self.X[self.X[:,i] == val]
                y_left = self.y[self.X[:,i] == val]
                X_right = self.X[self.X[:,i] != val]
                y_right = self.y[self.X[:,i] != val]
                if(X_left.shape[0] >= min_sam and X_right.shape[0] >= min_sam):
                    temp_left = Node(X_left, y_left, self.features)
                    temp_right = Node(X_right, y_right, self.features)
                else:
                    continue
                # print(self.features[i] + "==" + str(val) + " Info gain: "+ str(self.info_gain(temp_left, temp_right)))
                if(self.info_gain(temp_left, temp_right) > max_info_gain):
                    self.is_cat = True
                    max_info_gain = self.info_gain(temp_left, temp_right)
                    # print(self.features[i], val,max_info_gain)
                    self.left = temp_left
                    self.right = temp_right
                    self.criteria = self.features[i]
                    self.value = val
        else:
            values = []
            for i in range(self.X.shape[1]):
                vals = self.X[:,i]
                if(len(vals)>num_splits):
                    temp_vals = random.sample(list(vals), k=num_splits)
                    for x in temp_vals:
                        values.append((i, x))
                else:
                    for x in vals:
                        values.append((i, x))
            for i, val in set(values):
                X_left = self.X[self.X[:,i] <= val]
                y_left = self.y[self.X[:,i] <= val]
                X_right = self.X[self.X[:,i] > val]
                y_right = self.y[self.X[:,i] > val]
                if(X_left.shape[0] >= min_sam and X_right.shape[0] >= min_sam):
                    temp_left = Node(X_left, y_left, self.features)
                    temp_right = Node(X_right, y_right, self.features)
                else:
                    continue
                # print(self.features[i] + "<=" + str(val) + " Info gain: "+ str(self.info_gain(temp_left, temp_right)))
                if(self.info_gain(temp_left, temp_right) > max_info_gain):
                    self.is_cat = False
                    max_info_gain = self.info_gain(temp_left, temp_right)
                    self.left = temp_left
                    self.right = temp_right
                    self.criteria = self.features[i]
                    self.value = val
        if(self.criteria is not None):
            self.leaf = False
            self.X = None
            self.y = None
