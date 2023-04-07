# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:21:22 2023

@author: Jose Lopez
"""

import numpy as np


data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Print out some data points
print ('First 10 examples from the dataset:\n,')
for i in range(10):
    print ('x =', X[i, ], ', y =', y[i])
    
    def feature_normalize(X):
    
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0, ddof=1)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma


X, mu, sigma = feature_normalize(X)
X = np.hstack((np.ones((m, 1)), X))

# Choose some alpha value
alpha = 0.15
num_iters = 400

# Init theta and run gradient descent
theta = np.zeros(3)

def compute_cost_multi(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Linear regression parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in X and y.
    """
    m = len(y)
    diff = X.dot(theta) - y
    J = 1.0 / (2 * m) * diff.T.dot(diff)
    return J

def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Initial linear regression parameter.
    alpha : float
        Learning rate.
    num_iters: int
        Number of iteration.

    Returns
    -------
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    J_history: ndarray, shape (num_iters,)
        Cost history.
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta -= alpha / m * ((X.dot(theta) - y).T.dot(X))
        J_history[i] = compute_cost_multi(X, y, theta)

    return theta, J_history

theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

import matplotlib.pyplot as plt

print ('Theta computed from gradient descent:')
print (theta)


plt.figure()
plt.plot(range(1, num_iters + 1), J_history, color='b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

normalize_test_data = ((np.array([1650, 3]) - mu) / sigma)
normalize_test_data = np.hstack((np.ones(1), normalize_test_data))
price = normalize_test_data.dot(theta)
print ('Predicted price of a 1650 sq-ft, 3 br house:', price)

data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))

def normal_eqn(X, y):
    """
    Computes the closed-form solution to linear regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.

    Returns
    -------
    theta : ndarray, shape (n_features,)
        The closed-form solution to linear regression using the normal equations.
    """
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

theta = normal_eqn(X, y)
print ('Theta computed from the normal equations: ')
print (theta)

price = np.array([1, 1650, 3]).dot(theta)
print ('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price)