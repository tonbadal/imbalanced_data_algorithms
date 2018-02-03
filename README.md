# Some Algorithms for Imbalanced Data

Here you can find some algorithms to handle the problem of imbalanced data. It is has been assumed for some situations that the target class is the small class.

## K-Clustering

The K-Clustering algorithm clusters the big class into K separate clusters, and K classifiers are trained for each cluster against the target data. To classify a sample as positive (small), all the classifiers must agree on classifying the sample as positive.

The algorithm is follows these steps:
1. Find K clusters in the large class. The number of clusters K is defined so that each cluster contains a similar number of samples to the target class.
2. K classifiers are trained for each cluster data against the small class.
3. In the prediction stage, each sample is shown to the K classifiers. For a sample to be predicted as positive, all K classifiers must agree, otherwise is classified as negative.


This image shows an example where the large class is formed by three Gaussian Distributions. The base estimator showed in this example is logistic regression, and the three black lines show the decision boundaries for each classifier.


![alt text](https://github.com/tonbadal/imbalanced_data_algorithms/blob/master/kclustering.png)



## Dagging

Dagging stands for Down-sampling for Bagging. This algorithm randomly splits the samples of the big class into K chunks, and K classifiers are trained with each chunk against the whole target samples. The final classification is the geometric mean of the classification probabilities for all trained classifiers.


(More detailed description on progress)
