import logging

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA, SparsePCA
from sklearn.utils import resample


def text2num(text):
    """
    To be applied for a pandas DataFrame, to locate the missing values and convert them to NaN, e.g.
        > df['Column_str'] = df['Column'].apply(text2num)

    :param text: Text to process
    :return: 0 or input text without spaces on the edges
    """
    try:
        return np.float(text)
    except ValueError:
        return np.NaN
    except TypeError:
        return np.NaN


def text2class_equifax(text):
    """
    To be applied for a pandas DataFrame, to convert a column to 'kind of missing' string, e.g.
        > df['Column_str'] = df['Column'].apply(text2class)

    :param text: Text to process
    :return: 0 or input text without spaces on the edges
    """
    try:
        _ = np.float(text)
        return 0
    except ValueError:
        return text.strip()
    except TypeError:
        return text.strip()


def text2class_equifax(text):
    """
    To be applied for a pandas DataFrame, to convert a column to 'kind of missing' string, e.g.
        > df['Column_str'] = df['Column'].apply(text2class)

    :param text: Text to process
    :return: 0 or input text without spaces on the edges
    """
    try:
        _ = np.float(text)
        return 0
    except ValueError:
        return text.strip()
    except TypeError:
        return text.strip()


def str2onehot_df(df, label, delete_original=True, n_min=10):
    """
    Calculates the one-hot encoding representation of a given DataFrame's label and transforms the DataFrame.

    :param df: DataFrame to convert
    :param label: Column that will be converted to one-hot encoding
    :param delete_original: Whether to delete the original column
    :return: An array containing the names of the OneHot columns
    """
    classes = df[label].unique()
    # Take away the 0, it will be returned as a 'all zeros' vector
    classes = list(filter(lambda x: x != 0, classes))

    zeros = np.zeros(len(df))

    new_labels = []

    for c in classes:
        elems = len(df.loc[df[label] == c])
        if elems > n_min:
            df['{0}_OneHot_{1}'.format(label, c)] = pd.DataFrame(zeros, dtype='int8')
            df.loc[df[label] == c, '{0}_OneHot_{1}'.format(label, c)] = 1
            new_labels.append('{0}_OneHot_{1}'.format(label, c))
        else:
            logging.debug('{0}_OneHot_{1} discarded as it only has {2} samples'.format(label, c, elems))

    if delete_original:
        del df[label]

    return new_labels


def print_results(prob, y):
    """
    Print some metrics about a prediction

    :param f: predicted probabilities
    :param y: real labels
    :return:
    """

    f = np.array([0 if x < 0.5 else 1 for x in prob])

    print('\tAccuracy: {0:.2f} %'.format(accuracy_score(y, f) * 100))
    print('\tRecall: {0:.2f} %'.format(recall_score(y, f) * 100))
    print('\tTrue Pos Rate: {0:.2f} %'.format(sum((y == 1) & (f == 1)) / sum(f == 1) * 100))
    print('\tArea Under ROC: {0:.2f} %'.format(roc_auc_score(y, prob) * 100))
    print('\tF1: {0:.2f} %'.format(f1_score(y, f) * 100))
    print('\tCorrectly predicted 1s: {0}'.format(sum((y == 1) & (f == 1))))
    print('\tPredicted 1s: {0}'.format(sum(f)))
    print('\tReal 1s: {0}'.format(sum(y)))
    print('\tElements: {0}'.format(len(y)))


def reduce_dimensionality_df(df, labels, ndim=200, new_labels_prefix=None):
    """
    Apply PCA to reduce the dimensionality of a given DataFrame. It will only affect those columns given in 'labels'.

    :param df: pd.DataFrame where the dimensionality reduction will be applied
    :param labels: Names of those columns to reduce dimensionality
    :param ndim: Output dimensionality
    :param new_labels_prefix: The name of the new columns will be 'new_labels_prefix_PCA_N'
    :return: An array containing the names of the new columns
    """
    pca = PCA(n_components=ndim)
    X = pca.fit_transform(df[labels])

    for label in labels:
        del df[label]

    new_labels = []
    for i in range(ndim):
        if new_labels_prefix:
            label_name = '{0}_PCA_{1}'.format(new_labels_prefix, i)
        else:
            label_name = 'PCA_{1}'.format(new_labels_prefix, i)

        df[label_name] = pd.DataFrame(X.T[0], dtype='float32')
        new_labels.append(label_name)

    return new_labels


def reduce_size_dtype(df):
    for i in range(len(df.columns)):
        if df.dtypes[i] == np.int64:
            if max(df[df.columns[i]]) < 128:
                df[df.columns[i]] = pd.DataFrame(df[df.columns[i]], dtype='int8')
            elif max(df[df.columns[i]]) < 32767:
                df[df.columns[i]] = pd.DataFrame(df[df.columns[i]], dtype='int16')
            elif max(df[df.columns[i]]) < 2147483647:
                df[df.columns[i]] = pd.DataFrame(df[df.columns[i]], dtype='int32')

        elif df.dtypes[i] == np.float64:
            threshold = lambda x : 0 if x < 1E-10 else x
            df[df.columns[i]] = pd.DataFrame(df[df.columns[i]].apply(threshold), dtype='float32')



def upsample_data(df, down=False, bad_proportion=0.5, random_state=np.random.RandomState(seed=123)):
    df_bad = df[df['Bad'] == 1]
    df_good = df[df['Bad'] == 0]


    if not down:
        new_bad_len = round(bad_proportion / (1 - bad_proportion) * len(df_good))
        df_bad = resample(df_bad, replace=True, n_samples=new_bad_len, random_state=random_state)
        df = pd.concat([df_good, df_bad])

    elif down:
        new_good_len = round((1 - bad_proportion) / bad_proportion * len(df_bad))
        df_good = resample(df_good, replace=False, n_samples=new_good_len, random_state=random_state)
        df = pd.concat([df_good, df_bad])

    return df.sample(frac=1, random_state=random_state)


def print_proportions(df):
    print('BAD samples: {0} ({1:.2f} %)'.format(sum(df['Bad'] == 1), sum(df['Bad'] == 1) / len(df) * 100))
    print('GOOD samples: {0} ({1:.2f} %)'.format(sum(df['Bad'] == 0), sum(df['Bad'] == 0) / len(df) * 100))
    ratio = sum(df['Bad'] == 0) / sum(df['Bad'] == 1)
    print('Ratio GOOD/BAD: {0:.2f}'.format(ratio))
    print()


def voting(prob_arr, threshold = 0.5):
    out_len = len(prob_arr[0])
    n_voters = len(prob_arr)
    y = np.zeros(out_len)
    for i in range(out_len):
        vote = 0
        for k in range(n_voters):
            if prob_arr[k][i] > threshold:
                vote += 1

        if vote >= round(n_voters/2):
            y[i] = 1

    return y

