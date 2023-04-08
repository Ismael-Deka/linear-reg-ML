import pandas as pd
import random


df = pd.read_csv("data/data.csv")

rows = df.shape[0]

X_train = df['sqft_lot'].loc[:rows*0.75]
y_train = df['price'].loc[:rows*0.75]

X_test = df['sqft_lot'].loc[rows*0.75+1:]
y_test = df['price'].loc[rows*0.75+1:]

