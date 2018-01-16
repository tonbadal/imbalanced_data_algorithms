import logging
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import utils

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.DEBUG)


class DaggingClassifier:
    """
    Downsample for Bagging = Dagging
    """
    def __init__(self,
                 base_estimator=None,
                 fold_proportion=1.0,
                 overlapping=0.0):

        if base_estimator is None:
            self.base_estimator = LinearSVC()
        else:
            # self.base_estimator = clone(base_estimator)
            self.base_estimator = base_estimator

        assert 0 <= fold_proportion <= 1, 'Fold Proportion value must be 0 <= fold_proportion < 1'
        self.fold_proportion = fold_proportion

        assert 0 <= overlapping < 1, 'Overlapping value must be 0 <= overlapping < 1'
        self.overlapping = overlapping

        self.classifiers = dict()

    def fit(self, X, y):
        """Compute k-clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        """
        X = np.array(X)
        y = np.array(y)

        if len(X) != len(y):
            raise AttributeError('Sizes of X and y must be equal')

        targets = np.unique(y)
        if len(targets) > 2:
            raise NotImplementedError('Only available for two classes classification')

        # Number of samples of each class
        targets_len = np.array([len(y[y == target]) for target in targets])

        small_target_arg = np.argmin(targets_len)
        self.small_target_label = targets[small_target_arg]
        small_target_len = targets_len[small_target_arg]#= len(y[y == self.small_target_label])

        large_target_arg = np.argmax(targets_len)
        self.large_target_label = targets[large_target_arg]
        large_target_len = targets_len[large_target_arg]#= len(y[y == self.large_target_label])

        X_big = X[y == self.large_target_label]
        y_big = y[y == self.large_target_label]

        X_small = X[y == self.small_target_label]
        y_small = y[y == self.small_target_label]

        for X_c, i in self.chunks(X_big, n=int(round(small_target_len / self.fold_proportion)), overlapping=self.overlapping):
            y_c = self.large_target_label * np.ones(len(X_c))

            X_to_predict = np.concatenate((X_c, X_small))
            y_to_predict = np.concatenate((y_c, y_small))

            clf = clone(self.base_estimator)
            # clf = LogisticRegression()

            clf.fit(X_to_predict, y_to_predict)
            self.classifiers[i] = clf

    def predict_proba(self, X):
        """Prediction

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        X = np.array(X)

        f = np.ones(len(X))
        for k, clf in self.classifiers.items():
            if hasattr(clf, 'predict_proba'):
                f_c = clf.predict_proba(X).T[1]
            else:
                f_c = clf.predict(X)

            # f_c = [1 if xx > self.threshold else 0 for xx in f_c]
            f *= f_c
            # f += f_c

        f = np.sqrt(f)
        # f = f / len(self.classifiers)
        return np.vstack((1 - f, f)).T

    def predict(self, X):
        return self.predict_proba(X).T[1]

    def chunks(self, l, n, overlapping=0.0):
        """Yield successive n-sized chunks from l."""

        start = 0
        end = n
        i = 0
        while True:
            yield l[start:end], i
            if end == None:
                break
            i += 1
            start += int(np.floor(n * (1-overlapping)))
            if start + 2 * n <= len(l):
                end = start + n
            else:
                end = None

    def score(self, y, f):
        return roc_auc_score(y, f)

    def save_model(self, file_name='pickles/kclustering_model.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(list(self.classifiers.values()), f)

    def load_model(self, file_name='pickles/kclustering_model.pkl'):
        with open(file_name, 'rb') as f:
            classifiers = pickle.load(f)

        for k in range(len(classifiers)):
            self.classifiers[k] = classifiers[k]

    def get_variable_importances(self):
        if type(self.classifiers[0]) != DecisionTreeClassifier:
            raise NotImplementedError

        w = np.zeros(self.classifiers[0].n_features_)
        for c in self.classifiers.values():
            w += c.feature_importances_

        return w / len(self.classifiers)

    def __str__(self):
        return 'KClustering(base_estimator={0}, \nclustering_prop={1}, threshold={2}):'.format(
            self.base_estimator, self.clustering_prop, self.threshold)

if __name__ == "__main__":
    X = np.array([
        [1,     3.25],
        [1.5,   3],
        [2,     3.25],
        [1.5,   3.25],
        [3,     2],
        [3.25,  1.5],
        [3,     1],
        [3.25,  1],
        [1,     1],
        [1.25,  1.5],
        [2,     2]
    ])
    y = np.array([0,0,0,0,0,0,0,0,1,1,1])
    dc = DaggingClassifier(overlapping=0.0)
    dc.fit(X,y)

    f = dc.predict(X)
    print('## FINAL Prediction ##')
    utils.print_results(f, y)








