import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pickle

data = pd.read_excel('100mrun.xlsx')
lis_dat = list(data)
print(lis_dat)
sns.scatterplot(data=data, x=data['Year '] ,y=data['Time(sec)'])

data.corr()

x = data[['Year ']]
y = data['Time(sec)']

lr = LinearRegression()
lr.fit(x,y)

ypred1 = lr.predict(x)
pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb')) 
print(model.predict([[1996]]))
print(ypred1)
print(lr.intercept_)
print(lr.coef_)
print(mean_squared_error(ypred1,y))
print(r2_score(y,ypred1))
