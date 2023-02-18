import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("Your Machine Learning Platform")
st.write("#### Version 0.0")

video_url = 'https://www.youtube.com/watch?v=Klqn--Mu2pE'
st.write("- Up to this point I followed the tutorial on [this video](%s),\
          but as always I have some ideas to go beyond" % video_url)

st.markdown("""
- Next steps:
  - Code refactoring
  - New dataset option: create your own with a set of parameters
  - More options to choose what to see in the plot (not necessarily PCA)
  - More information about the algorithm
  - More parameters to set
  - Other types of algorithms (clustering, regression)
  - Section of resources to learn about the algorithm
""")

st.subheader("Algorithm Info")

dataset_name = st.sidebar.selectbox("Select Dataset", ('Iris', 'Breast Cancer', 'Wine'))

classifier_name = st.sidebar.selectbox("Select Classifier", ('KNN', 'SVM', 'Random Forest'))

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y#, data.DESCR

X, y = get_dataset(dataset_name)

st.write("shape of dataset:", X.shape)
st.write("number of classes:", len(np.unique(y)))

def add_parameter_ui(classifier_name):
    params = dict()

    if classifier_name == 'KNN':
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif classifier_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == 'SVM':
        classifier = SVC(C=params['C'])
    else:
        classifier = RandomForestClassifier(n_estimators=params['n_estimators'],
                                            max_depth=params['max_depth'],
                                            random_state=150223)
    
    return classifier

classifier = get_classifier(classifier_name, params)

# classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150223)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar()

st.pyplot(fig)