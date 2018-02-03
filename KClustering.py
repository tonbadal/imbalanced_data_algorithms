import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from sklearn.metrics import roc_auc_score

import utils

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.DEBUG)


class KClustering:
    def __init__(self,
                 base_estimator=None,
                 final_estimator=None,
                 clustering_prop=1,
                 threshold=0.5):

        if base_estimator is None:
            self.base_estimator = LogisticRegression()
        else:
            # self.base_estimator = clone(base_estimator)
            self.base_estimator = base_estimator

        self.final_estimator = final_estimator

        self.clustering_prop = clustering_prop
        self.threshold = threshold

        self.n_clusters = None
        self.classifiers = dict()

    def fit(self, X, y, plot=False):
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

        if not self.n_clusters:
            self.n_clusters = int(round(large_target_len / small_target_len * self.clustering_prop))
        logging.info('{0} clusters used'.format(self.n_clusters))

        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                         precompute_distances='auto', verbose=0, random_state=None, copy_x=True,
                         n_jobs=1, algorithm='auto')

        X_clusters = X[y == self.large_target_label]
        y_clusters = kmeans.fit_predict(X_clusters)

        X_small = X[y == self.small_target_label]
        y_small = y[y == self.small_target_label]

        for i in range(self.n_clusters):
            X_c = X_clusters[y_clusters == i]
            y_c = self.large_target_label * np.ones(sum(y_clusters == i))

            X_to_predict = np.concatenate((X_c, X_small))
            y_to_predict = np.concatenate((y_c, y_small))

            clf = clone(self.base_estimator)
            # clf = LogisticRegression()

            clf.fit(X_to_predict, y_to_predict)
            self.classifiers[i] = clf

            if plot and type(clf) != LogisticRegression:
                logging.error('Plotting option only available for Logistic Regression.')

            elif plot and type(clf) == LogisticRegression:
                if i == 1:
                    sns.kdeplot(X_c.T[0], X_c.T[1], cmap="Reds", shade=True, shade_lowest=False)
                    sns.kdeplot(X_small.T[0], X_small.T[1], cmap="Blues", shade=True, shade_lowest=False)
                else:
                    sns.kdeplot(X_c.T[0], X_c.T[1], cmap="Reds", shade=True, shade_lowest=False)

                m = - clf.coef_[0][0] / clf.coef_[0][1]
                n = - clf.intercept_[0] / clf.coef_[0][1]

                x1 = np.max((np.min(X_c.T[0]), np.min(X_small.T[0])))
                x2 = np.min((np.max(X_c.T[0]), np.max(X_small.T[0])))
                y1 = m * x1 + n
                y2 = m * x2 + n
                plt.plot([x1, x2], [y1, y2], 'k', linewidth=1)

        plt.scatter(X.T[0], X.T[1], c=y, cmap="RdBu", edgecolor="white", vmin=-0.1, vmax=1.1)
        plt.show()

    def predict(self, X):
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

            f_c = [1 if xx > self.threshold else 0 for xx in f_c]
            f *= f_c

        return np.sqrt(f)

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


def test():
    m1, m2, m3, mt = ([-4, 2], [0, 5], [4, 2], [0, 0])
    C = np.array([[1, 0], [0, 1]])

    n_samples = 1000

    X1 = np.random.multivariate_normal(m1, C, n_samples)
    X2 = np.random.multivariate_normal(m2, C, n_samples)
    X3 = np.random.multivariate_normal(m3, C, n_samples)
    Xt = np.random.multivariate_normal(mt, C, n_samples)

    X = np.vstack((X1, X2, X3, Xt))
    y = np.concatenate((np.zeros(3 * n_samples), np.ones(n_samples)))

    kc = KClustering()

    kc.fit(X, y, plot=True)

    f = kc.predict(X)
    print('## Prediction Results ##')
    utils.print_results(f, y)

if __name__ == "__main__":
    test()









