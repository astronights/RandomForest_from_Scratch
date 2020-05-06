from abc import ABC

from collections import Counter

import math
import random
import numpy as np
import pandas as pd

from rf_node import Node
from rf_tree import Tree


class RandomForestClassifier(ABC):

    def __init__(self, n_iter=100, n_feat = 0.5, metric='gini',
                 max_depth=6, min_samples_split = 2, num_splits=5,
                 min_samples_leaf = 1, num_bootstrap_samples = 1.0):
        self.n_iter = n_iter
        self.n_feat = n_feat
        self.metric = metric
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_splits = num_splits
        self.predictions = None
        self.features = None
        self.target = None
        self.trees = []
        self.scaler = []


    def fit(self, X_train, y_train):
        self.features = X_train.columns.values
        self.target = y_train.columns.values[0]
        if(y_train[self.target].dtype == 'object'):
            self.max_depth *= 2
        X_train, y_train = self._to_numpy(X_train), self._to_numpy(y_train)
        for i in range(self.n_iter):
            if(i%10 == 0):
                print("Iterations: ", i)
            X_boot, y_boot = self._bootstrap(X_train, y_train)
            tree = self._grow_decision_tree(X_boot, y_boot, self.features)
            # tree.print_tree()
            self.trees.append(tree)
        print("Random Forest has been fit.")

    def _calc_param(self, param, size):
        if(isinstance(param, int)):
            res = param
        else:
            res =(math.floor(param*size))
        return(res)

    def _bootstrap(self, X_train, y_train):
        val, freq = np.unique(y_train, return_counts=True)
        weights = (dict(zip(val, len(y_train)/freq)))
        res_weights = (np.vectorize(weights.get)(y_train))
        ix = random.choices(list(np.arange(X_train.shape[0])),
                            weights=res_weights,
                            k=self._calc_param(self.num_bootstrap_samples, X_train.shape[0]))
        return(X_train[ix], y_train[ix])
        # return(X_train.iloc[ix,:], y_train.iloc[ix,:])

    def _grow_decision_tree(self, X_boot, y_boot, features):
        X, features = self._feature_selection(X_boot, features)
        y = y_boot
        node = Node(X, y, features, self.metric)
        tree = Tree(node, self.max_depth, self.min_samples_split, self.min_samples_leaf, self.num_splits)
        # tree.print_tree()
        tree.build_tree()
        return(tree)

    def _feature_selection(self, X, features):
        ix = random.sample(list(np.arange(len(X[0]))), \
                           k=self._calc_param(self.n_feat, X.shape[1]))
        # ix = random.sample(list(np.arange(X.shape[1])), \
                           # k=math.floor(self.n_feat*X.shape[1]))
        # return(X.iloc[:,ix])
        return(X[:,ix], features[ix])

    def _bag_pred(self, X):
        final = []
        for tree in self.trees:
            final.append(tree.predict(X))
        c = Counter(final)
        return(c.most_common(1)[0][0])

    def _to_numpy(self, data):
        # v = data.reset_index()
        # np_res = np.rec.fromrecords(v, names=v.columns.tolist())
        # return(np_res)
        res = data.to_numpy()
        try:
            if(res.shape[1] == 1):
                res = res.flatten()
        except:
            pass
        return(res)

    def predict(self, X):
        y_pred = pd.DataFrame(columns=[self.target], index=X.index)
        y_pred = X.apply(self._bag_pred, axis=1)
        return(self._to_numpy(y_pred))

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = self._to_numpy(y)
        accuracy = np.sum(y_pred == y_true)
        return(accuracy/len(y_true))

    def confusion(self, X, y):
        y_pred = self.predict(X)
        y_true = self._to_numpy(y)
        df_confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        print()
        print(df_confusion)

    def metrics(self, X, y):
        y_pred = self.predict(X)
        y_true = self._to_numpy(y)
        res = pd.DataFrame(columns=['Precision', 'Recall', 'F1-Score'])
        trues = y_pred[y_pred == y_true]
        for val in set(y_true):
            tp = np.sum(trues == val)
            tp_fp = np.sum(y_pred == val)
            tp_fn = np.sum(y_true == val)
            prec = tp/tp_fp
            rec = tp/tp_fn
            f1 = (2*prec*rec)/(prec+rec)
            res = res.append(pd.Series({'Precision':prec, 'Recall':rec, 'F1-Score': f1}, name=val))
        print()
        print(res)
