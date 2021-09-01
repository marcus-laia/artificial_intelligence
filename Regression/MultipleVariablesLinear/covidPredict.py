import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn import linear_model as lm

### LOAD DATA ###
print('-'*30);print('HEAD');print('-'*30)

data = pd.read_csv('../resources/coronaCases.csv',sep=',')
data = data[['id','cases']]
print(data.head())

### PREPARE DATA ###
print('-'*30);print('PREPARE DATA');print('-'*30)

x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')

polyFeat = pf(degree=3)
x = polyFeat.fit_transform(x)
print(x)

### TRAINING DATA ###
print('-'*30);print('TRAINING DATA');print('-'*30)
model = lm.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
# print(f'Coefficients: {model.coef_}%')
# print(f'Intercept: {model.intercept_}%')
print(f'Accuracy: {round(accuracy*100,3)}%')

y0 = model.predict(x)

### PREDICTION DATA ###
print('-'*30);print('PREDICTION');print('-'*30)

days = 30
print(f'Prediction - Cases after {days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[234+days]])))/1000000,2),'Million')

x1 = np.array(list(range(1,234+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))

plt.plot(y1,'--r')
plt.plot(y0,'--b')
plt.show()