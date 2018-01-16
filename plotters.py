import logging

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist_df(var, df, plot_norm=True):
    """
    Plots a variable from a given dataframe and compares both 'Bad' options with a gaussian distribution
    :param var:
    :param df:
    :param plot_norm:
    :return:
    """
    if var not in np.concatenate([df.select_dtypes('int64').columns, df.select_dtypes('float64').columns]):
        return 0

    h, x = np.histogram(df.loc[df['Bad'] == 0, var], density=True)
    x = [(x[i] + x[i + 1]) / 2 for i in range(len(x[:-1]))]
    plt.plot(x, h, 'g')
    if plot_norm:
        plt.plot(x, norm.pdf(x, np.mean(df.loc[df['Bad'] == 0, var]), np.std(df.loc[df['Bad'] == 0, var])), 'g--')
    plt.plot()

    h, x = np.histogram(df.loc[df['Bad'] == 1, var], density=True)
    x = [(x[i] + x[i + 1]) / 2 for i in range(len(x[:-1]))]
    plt.plot(x, h, 'r')
    if plot_norm:
        plt.plot(x, norm.pdf(x, np.mean(df.loc[df['Bad'] == 1, var]), np.std(df.loc[df['Bad'] == 1, var])), 'r--')
    plt.title(var)

    #plt.hist(df.loc[df['Bad'] == 0, var], bins=10, rwidth=0.8, color='blue', density=True, histtype='step')
    #plt.hist(df.loc[df['Bad'] == 1, var], bins=10, rwidth=0.8, color='red', density=True, histtype='step')
    plt.show()
    return 1


def plot_distributions_df(df, labels, title=None):
    """
    For each label in 'labels', plot the distributions of good samples and bad samples

    :param df: DataFrame
    :param labels:
    :param title: Title of the plot
    :return:
    """
    for label in labels:
        try:
            sns.distplot(df.loc[df['Bad'] == 0, label], bins=50)
            sns.distplot(df.loc[df['Bad'] == 1, label], bins=50)
            plt.title(title + 'Feature_' + label)
            plt.show()
        except:
            logging.warning('Could not plot %s' % label, exc_info=True)




def plot_hist(idx, X, df, plot_norm=True):
    """
    Plots an index from a given matrix and compares both 'Bad' options (from a dataframe) with a gaussian distribution

    :param var:
    :param df:
    :param plot_norm:
    :return:
    """
    X = X.T[idx]
    h, x = np.histogram(X[df['Bad'] == 0], density=True)
    x = [(x[i] + x[i + 1]) / 2 for i in range(len(x[:-1]))]
    plt.plot(x, h, 'g')
    if plot_norm:
        plt.plot(x, norm.pdf(x, np.mean(X[df['Bad'] == 0]), np.std(X[df['Bad'] == 0])), 'g--')

    h, x = np.histogram(X[df['Bad'] == 1], density=True)
    x = [(x[i] + x[i + 1]) / 2 for i in range(len(x[:-1]))]
    plt.plot(x, h, 'r')
    if plot_norm:
        plt.plot(x, norm.pdf(x, np.mean(X[df['Bad'] == 1]), np.std(X[df['Bad'] == 1])), 'r--')

    plt.show()
    return 1


def plot_roc(probs, y, title=None):
    """
    Given predict_probs and y_true, this function plots the ROC curve.

    :param probs:
    :param y:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='AUC = {0:.3f}'.format(area))
    plt.plot([0,0], [1,1], 'k:')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    return area


def plot_many_roc(probs, y, labels=[], title=None):
    """
    Given two tuples predict_probs and labels, and the real outputs, this function plots the ROC curves.

    :param probs:
    :param y:
    :return:
    """
    for i in range(len(probs)):
        fpr, tpr, thresholds = roc_curve(y, probs[i], pos_label=1)
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='{0}. AUC = {1:.3f}'.format(labels[i], area))
        plt.plot([0,1], [0,1], 'k:')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_logistic_weights(w, labels, n=None, title=None, plot_topn_labels=0):
    """
    Given an array of weights and an array of labels, a horizontal bar chart is ploted showing the respective weight of
    each label. For a large amount of labels, set 'n' to show only 2*n bars in the chart, the 'n' biggest and 'n'
    smallest weights/labels.

    :param w:
    :param labels:
    :param n:
    :param title:
    :param plot_topn_labels:
    :return:
    """
    if n and len(w) > 2*n > 0:
        w = np.concatenate((w[:n], w[-n:]))
        labels = np.concatenate((labels[:n], labels[-n:]))
    else:
        n = None

    sorted_idx = [i[0] for i in sorted(enumerate(abs(w)), key=lambda x:x[1])]
    w_sorted = [abs(w[i]) for i in sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]

    plot_topn_labels = min(len(w) - 1, plot_topn_labels)
    for i in range(1, plot_topn_labels + 1):
        print(labels_sorted[-i], ' :: ', w_sorted[-i])

    if n:
        w_sorted.insert(n, 0)
        labels_sorted.insert(n, '. . .')

    y_pos = range(len(w_sorted))
    plt.barh(y_pos, w_sorted, align='center', alpha=0.5)
    plt.yticks(y_pos, labels_sorted)
    if title:
        plt.title(title)
    plt.show()
