# Some Algorithms for Imbalanced Data

Here you can find some algorithms to handle the problem of imbalanced data. It is has been assumed for some situations that the target class is the small class.

## K-Clustering

The K-Clustering algorithm clusters the big class into K separate clusters, and K classifiers are trained for each cluster against the target data. To classify a sample as possitive (small), all the classifiers must agree on classifying the sample as positive.

(More detailed description on progress)

## Dagging

Dagging stands for Downsampling for Bagging. This algorithm randomly splits the samples of the big class into K chunks, and K classifiers are trained with each chunk against the whole target samples. The final classification is the geometric mean of the classification  probabilities for all trained classifiers.

(More detailed description on progress)
