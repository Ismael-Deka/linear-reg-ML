"""
Tests linear regression model for predicting home prices.

This script loads the test data from a pickle file, performs testing using a trained 
linear regression model, and prints the cost and mean percent difference between predicted 
and true values.

Usage:
    python test.py

Requirements:
    - The `pickle` module (imported as `pk`) must be installed.
    - The `lin_utils` module must be available in the same directory.

Note:
    Before running this script, make sure to run `train.py` to train a linear regression 
    model and generate the necessary test data.

"""
import pickle as pk
import sys
from lin_utils import test


try:
    with open('pickle/test.pkl', 'rb') as handle:
        X_test = pk.load(handle)
        y_test = pk.load(handle)
        weights = pk.load(handle)
        bias = pk.load(handle)

except FileNotFoundError:
    print("Test dataset not found. Please run train.py before running test.")
    sys.exit()
y_test = y_test.where(y_test <= 1000000, 999999)
cost, mean_percent_diff = test(X_test=X_test, y_test=y_test, weights=weights, bias=bias)
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print(f"Cost: {cost}")
print(f"Mean Percent Difference: {mean_percent_diff}")
