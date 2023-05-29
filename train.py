"""
Train a linear regression model for predicting home prices.

This script reads data from a CSV file, preprocesses it by removing rows with zero prices 
and outliers, splits it into training and testing sets, and saves the necessary data for 
testing in a pickle file.

Usage:
    python train.py

Requirements:
    - The `os` module must be imported.
    - The `pickle` module (imported as `pk`) must be installed.
    - The `pandas` module (imported as `pd`) must be installed.
    - The `lin_utils` module must be available in the same directory.

Note:
    Before running this script, make sure to have a CSV file named 'data.csv' in the same 
    directory with the appropriate data columns.

"""
import os
import pickle as pk
import pandas as pd
from lin_utils import train


df = pd.read_csv("data/data.csv")

df = df.drop(df[df['price'] == 0.0].index) # Remove rows where home has no price

df= df.sample(frac=1).reset_index(drop=True)

df['price'] = df['price'].where(df['price'] <= 1000000, 999999) # Remove outliers

# Calculate average price for each 'statezip'
avg_prices = df.groupby('statezip')['price'].mean()

# Assign the average prices to the corresponding rows in 'df'
df['avgprice'] = df['statezip'].map(avg_prices)

rows = df.shape[0]

train_split = int(rows * 0.80)
test_split = int(rows * 0.80) + 1

X_train = df[['sqft_lot', 'sqft_living', 'bathrooms', 'bedrooms', 'condition', 'avgprice']].loc[:train_split]
y_train = df['price'].loc[:train_split]
X_test = df[['sqft_lot', 'sqft_living', 'bathrooms', 'bedrooms', 'condition', 'avgprice']].loc[test_split:]
y_test = df['price'].loc[test_split:]


n_features = X_train.shape[1]

weights, bias = train(X_train, y_train, n_features)


if os.path.exists("pickle") is not True:
    os.mkdir("pickle")

with open('pickle/test.pkl', 'wb') as handle:
    pk.dump(X_test, handle)
    pk.dump(y_test, handle)
    pk.dump(weights, handle)
    pk.dump(bias, handle)
