import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from model_interface import Model


class Classification(Model):
    def __init__(self) -> None:
        self.available_datasets = ("Iris", "Breast Cancer", "Wine")

        self.dataset_name = None
        self.X = None
        self.y = None
        self.data_desc = None

        self.runners = {
            "KNN": KNeighborsClassifier,
            "SVM": SVC,
            "Random Forest": RandomForestClassifier,
        }

        self.model_name = None
        self.params = None

        self.sidebar = {
            "KNN": {
                "slider": [
                    {"label": "n_neighbors", "min_value": 1, "max_value": 15, "value": 3}
                ]
            },
            "SVM": {
                "slider": [
                    {"label": "C", "min_value": 0.01, "max_value": 10.0, "value": 5.0}
                ]
            },
            "Random Forest": {
                "slider": [
                    {"label": "n_estimators", "min_value": 1, "max_value": 100, "value": 10},
                    {"label": "max_depth", "min_value": 2, "max_value": 15, "value": 8},
                ]
            },
        }

    def get_available_datasets(self):
        return self.available_datasets

    def get_available_models(self):
        return list(self.runners.keys())

    def set_dataset(self, dataset_name):
        self.dataset_name = dataset_name

        data_reader = f"load_{self.dataset_name.lower().replace(' ', '_')}"
        dataset = getattr(datasets, data_reader)()

        # Study caching. Maybe it's worth to store X, y and desc as attributes
        #   This way, the dataset needs to be read and processed only when it's changed itself
        #   If this isn't possible, it's more reasonable to define X and y before fitting the model

        # features and targets for the model
        self.X, self.y = dataset.data, dataset.target

        # dataset description
        self.data_desc = dataset.DESCR

    def set_model(self, model_name):
        self.model_name = model_name

    def get_sidebar(self):
        assert self.model_name is not None, "Model Name not assigned"
        return self.sidebar[self.model_name]

    def set_params(self, params):
        self.params = params

    def get_model_info(self):
        assert self.model_name is not None, "Model Name not assigned"

        model = self.runners[self.model_name]()
        model.set_params(**self.params)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=150223
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        return f"""
        - Shape of input data: {self.X.shape}
        - Number of classes: {len(self.y)}
        - Classifier: {self.model_name}
        - Model Accuracy: {acc}
        """

    def get_dataset_info(self):
        return self.data_desc

    def get_figure(self):
        pca = PCA(2)
        X_projected = pca.fit_transform(self.X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2, c=self.y, alpha=0.8, cmap="viridis")

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.colorbar()

        return fig
