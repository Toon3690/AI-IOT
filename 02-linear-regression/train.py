import sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#for visualizing
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Load the data from 'data/housing.csv' into a pandas frame called housing
def load_housing_data():
    csv_path = os.path.join('./data', "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()


#Start

df = housing.drop(columns=['total_bedrooms', 'ocean_proximity'])
df_x = df.drop(columns=['median_house_value'])
df_y = df.median_house_value

X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, Y_train)

pr = model.predict(X_test)
print('Prediction first value:', pr[0])
print('for the second', pr[1])
print('for the third', pr[2])
print('for the fourth', pr[3])
print('for the fifth', pr[4])

print(Y_test[0:5])
