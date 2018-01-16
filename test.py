import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from KClustering import KClustering

import utils


class MyTesting:
    def __init__(self):
        pass

    def plot_results(self, X, y):
        sns.jointplot(x=0, y=1, data=pd.DataFrame(X))

    def testing_simple(self):

        X = np.array([
            [1,3],
            [2,3],
            [3,2],
            [3,1],
            [1,1],
            [2,2]
        ])
        y = np.array([0, 0, 0, 0, 1, 1])
        kc = KClustering()
        kc.fit(X, y)

        utils.print_results(kc.predict(X), y)

        self.plot_results(X, y)

    def testing_fraud(self):
        df = pd.read_csv('creditcard.csv')
        df, _ = train_test_split(df, test_size=0.95)

        plt.scatter(df.loc[df['Class'] == 0, "V1"].values, df.loc[df['Class'] == 0, "V2"].values, 'b')
        plt.scatter(df.loc[df['Class'] == 1, "V1"].values, df.loc[df['Class'] == 1, "V2"].values, 'r')

        plt.show()


if __name__ == "__main__":
    test = MyTesting()
    test.testing_fraud()