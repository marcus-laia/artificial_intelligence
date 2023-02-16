import numpy as np

class KNN:
    """K-Nearest Neighbors algorithm implementation"""
    def __init__(self, k=3) -> None:
        self.k = k

        self.fitted_x = None
        self.labels = None

    def fit(self, X, y):
        """Fit the trainning data."""
        self.fitted_x = X
        self.labels = y

    def predict(self, X):
        """Predict labels for new data."""
        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, sample):
        """Predict the label for a single sample."""
        # calculate distances
        distances = [self._eucliean_dist(sample, fitted_sample) for fitted_sample in self.fitted_x]
        # get the indexes and labels of the k neighbors with the smaller calculated distances
        k_nearest_indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = self.labels[k_nearest_indexes]
        # select the most common label
        sample_label = np.bincount(k_nearest_labels).argmax()

        return sample_label

    @staticmethod
    def _eucliean_dist(point_1, point_2):
        return np.sqrt(np.sum(np.square(point_1 - point_2)))
