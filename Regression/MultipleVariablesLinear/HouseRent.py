import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing, linear_model

# LOAD DATA
print('-'*30);print("IMPORTING DATA");print('-'*30)
data = pd.read_csv('../resources/houses_to_rent.csv', sep=',')
data = data[['city','rooms','bathroom','parking spaces','fire insurance',
             'furniture','rent amount']]
print(data.head())

# PROCESS DATA
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',','')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',','')))

le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform((data['furniture']))

print(data.head())

print('-'*30);print("CHECKING NULL DATA");print('-'*30)
print(data.isnull().sum())
# data = data.dropna()
# either replace with average or 0 or other constant

print('-'*30);print("HEAD DATA");print('-'*30)
print(data.head())

# SPLIT DATA
print('-'*30);print("SPLIT DATA");print('-'*30)

x = np.array(data.drop(['rent amount'],1))
y = np.array(data['rent amount'])
print('x',x.shape)
print('Y',y.shape)

xTrain, xTest, yTrain, yTest = sk.model_selection.train_test_split(x, y, test_size=0.2)
print('xTrain',xTrain.shape)
print('xTest',xTest.shape)

# TRAINING
print('-'*30);print("TRAINING");print('-'*30)

model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest,yTest)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Accuracy:', round(accuracy*100,3),'%')

# EVALUATION
print('-'*30);print("MANUAL TESTING");print('-'*30)

testValues = model.predict(xTest)
print(testValues.shape)

error=[]
for i,value in enumerate(testValues):
    error.append(yTest[i]-value)
    #print(f'Actual: {yTest[i]} Prediction: {int(value)} Error: {int(error[i])}')

