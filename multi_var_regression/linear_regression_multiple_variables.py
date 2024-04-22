import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("homeprices.csv")

# To fill the missing value, we calculate the median of the bedrooms column
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

# Create and train the model
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
print("The coefficients are: ", reg.coef_)
print("The intercept is: ", reg.intercept_)

# Make a prdiction
pred = reg.predict([[3000, 3, 40]])
print("The prediction for a 3000sqft, 3bdr, 40yr house is: ", pred)