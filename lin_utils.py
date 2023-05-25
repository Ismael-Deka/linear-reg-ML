"""
This module provides functions for linear regression training and evaluation.

Functions:
- mean_squared_error(y_true, y_pred): Calculates the mean squared error between true values 
  and predicted values.

- find_gradient(X, y, y_pred): Calculates the gradient for linear regression weights.

- find_bias_gradient(y, y_pred): Calculates the gradient for the bias term in linear regression.

- plot_results(y_true, y_pred, result_type): Plots the scatter plot of true values against predicted 
  values with a line of best fit.

- plot_history(history, title): Plots the history of a specific metric during training.

- train(X_train, y_train, n_features, learning_rate, momentum, num_iterations): Trains a linear 
  regression model using gradient descent.

- test(X_test, y_test, weights, bias): Test a linear regression model on  test set.
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score



def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error (MSE) between true values and predicted values.

    Parameters:
        y_true (array-like): Array or list of true values.
        y_pred (array-like): Array or list of predicted values.

    Returns:
        float: Mean squared error.
    """
    return (y_true - y_pred).pow(2).mean() * 0.5

def find_gradient(X, y, y_pred):
    """
    Calculates the gradient for linear regression weights.

    Parameters:
        X (array-like): Array or list of input features.
        y (array-like): Array or list of true values.
        y_pred (array-like): Array or list of predicted values.

    Returns:
        float: Gradient for the weights.
    """
    return (X.T * (y_pred - y)).mean()

def find_bias_gradient(y, y_pred):
    """
    Calculates the gradient for the bias term in linear regression.

    Parameters:
        y (array-like): Array or list of true values.
        y_pred (array-like): Array or list of predicted values.

    Returns:
        float: Gradient for the bias.
    """
    return (y_pred - y).mean()

def plot_results(y_true, y_pred, result_type):
    """
    Plots the scatter plot of true values against predicted values with a line of best fit.

    Parameters:
        y_true (array-like): Array or list of true values.
        y_pred (array-like): Array or list of predicted values.
        result_type (str): Type of results (Training or Testing).
    """

    plt.scatter(y_true, y_pred, s=15)

    m, b = np.polyfit(y_true, y_pred, deg=1)

    r_squared = r2_score(y_true, y_pred)

    plt.plot(y_true, m*y_true + b, color='red', label=f'R^2: {r_squared}')

    plt.legend()

    plt.title(f'{result_type} result')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.savefig(F"results/{result_type}/{result_type}_results.png")

    plt.clf()

def plot_history(history, title):
    """
    Plots the history of a specific metric during training.

    Parameters:
        history (list): List of metric values at each iteration.
        title (str): Title for the plot.
    """
    plt.plot(history)
    plt.title(f"{title} History")
    plt.xlabel('Iterations')
    plt.ylabel(title)

    plt.savefig(f"results/training/{title}_history.png")

    plt.clf()

def train(X_train, y_train, n_features, learning_rate=2.0, momentum=0.9, num_iterations=20000):

    """
    Trains a linear regression model using gradient descent.

    Parameters:
        X_train (pandas.DataFrame): DataFrame of input features.
        y_train (array-like): Array or list of true values.
        n_features (int): Number of input features.
        learning_rate (float): Learning rate for gradient descent (default=2.0).
        momentum (float): Momentum value for gradient descent (default=0.9).
        num_iterations (int): Maximum number of training iterations (default=20000).

    Returns:
        tuple: Tuple of trained weights and bias term.
    """

    weights = [random.uniform(-1, 1) for _ in range(n_features)]
    bias = random.uniform(-0.5, 0.5)

    y_pred = X_train @ weights + bias

    cost = mean_squared_error(y_train, y_pred)
    iteration = 1

    learning_rate_decay = 0.01
    momentum = 0.9
    grad = [0] * n_features

    mean_percent_diff = ((y_pred-y_train)/y_train*100).abs().mean()

    cost_history = [cost]
    mpd_history = [mean_percent_diff]


    while iteration <= num_iterations:
        weights_prev = weights[:]
        bias_prev = bias
        y_pred_prev = y_pred

        grad = [find_gradient(X_train.iloc[:, i], y_train, y_pred) +
                (grad[i] * momentum) for i in range(n_features)]


        weights = [weights[i] - grad[i] * learning_rate for i in range(n_features)]
        bias -= find_bias_gradient(y_train, y_pred) * learning_rate

        y_pred = X_train @ weights + bias

        cost = mean_squared_error(y_train, y_pred)

        mean_percent_diff = ((y_pred-y_train)/y_train*100).abs().mean()

        if cost > cost_history[-1]:
            weights = weights_prev
            bias = bias_prev
            y_pred = y_pred_prev

            if (learning_rate - learning_rate_decay) <= 0:
                learning_rate_decay /= 10

            learning_rate -= learning_rate_decay
            continue
        print(f"Iteration: {iteration}")
        print(f"Learn rate: {learning_rate}")
        print(f"Cost: {cost}")
        print(f"Weights: {weights}")
        print(f"Bias: {bias}")
        print("---------------------------------------------")

        cost_history.append(cost)
        mpd_history.append(mean_percent_diff)
        iteration += 1

    y_pred = y_pred.clip(upper=999999)
    mean_percent_diff = ((y_pred-y_train)/y_train*100).abs().mean()

    print(f"Weights: {weights}")
    print(f"Bias: {bias}")
    print(f"Mean Percent Difference: {mean_percent_diff}")

    if os.path.exists("results/training") is not True:
        os.makedirs("results/training")

    plot_results(y_true=y_train, y_pred=y_pred,result_type="training")
    plot_history(cost_history,"Cost")
    plot_history(mpd_history, "Mean Percent Diff")

    return weights, bias

def test(X_test, y_test, weights, bias):
    """
    Test a linear regression model on  test set.

    Parameters:
        X_test (pandas.DataFrame): DataFrame of input features for the test set.
        y_test (array-like): Array or list of true values for the test set.
        weights (list): List of trained weights.
        bias (float): Trained bias term.

    Returns:
        tuple: Tuple containing the cost (mean squared error) and mean percent 
        difference between predicted and true values.
    """

    y_pred = X_test @ weights + bias
    y_pred = y_pred.clip(upper=999999)
    cost = mean_squared_error(y_test, y_pred)


    mean_percent_diff = ((y_pred-y_test)/y_test*100).abs().mean()

    if os.path.exists("results/testing") is not True:
        os.makedirs("results/testing")

    plot_results(y_true=y_test, y_pred=y_pred,result_type="testing")

    return cost, mean_percent_diff
