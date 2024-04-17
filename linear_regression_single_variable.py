import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show()

reg = linear_model.LinearRegression()
# Fit the model
reg.fit(df[['area']], df.price)
# Run a prediction
y = reg.predict([[5000]])
print(f'Predicting for 3300', y)
w = reg.coef_
print(f'The model coefficient is: ', w)
b = reg.intercept_
print(f'The intercep values is: ', b)

assert y == w * 5000 + b

plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

d = pd.read_csv("areas.csv")
print(d.head(3))

p = reg.predict(d)
d['prices'] = p
print(d)