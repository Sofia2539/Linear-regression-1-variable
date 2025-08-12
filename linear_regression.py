#Import Packages & Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

#Problem - excersice
"""
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

You would like to expand your business to cities that may give your restaurant higher profits.
The chain already has restaurants in various cities and you have data for profits and populations from the cities.
You also have data on cities that are candidates for a new restaurant.
For these cities, you have the city population.
use the data to help you identify which cities may potentially give your business higher profits
"""

#Dataset
#The load_data() function shown below loads the data into variables x_train and y_train
#x_train is the population of a city
#y_train is the profit of a restaurant in that city. A negative value for profit indicates a loss.
#Both X_train and y_train are numpy arrays.
# load the dataset

# Compute Cost
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    """
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# Compute Gradient
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression.
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


# Gradient Descent
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

if __name__ == "__main__":
    # Load data
    x_train, y_train = load_data()

    print("Type of x_train:", type(x_train))
    print("First five elements of x_train are:\n", x_train[:5])
    print("Type of y_train:", type(y_train))
    print("First five elements of y_train are:\n", y_train[:5])
    print('The shape of x_train is:', x_train.shape)
    print('The shape of y_train is: ', y_train.shape)
    print('Number of training examples (m):', len(x_train))

    # Visualize data
    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title("Profits vs. Population per city")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

    # Initial cost check
    initial_w = 2
    initial_b = 1
    cost = compute_cost(x_train, y_train, initial_w, initial_b)
    print(f'Cost at initial w: {cost:.3f}')

    # Gradient descent
    initial_w = 0.
    initial_b = 0.
    iterations = 1500
    alpha = 0.01

    w, b, _, _ = gradient_descent(x_train, y_train, initial_w, initial_b,
                                  compute_cost, compute_gradient, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    # Predictions
    m = x_train.shape[0]
    predicted = np.zeros(m)
    for i in range(m):
        predicted[i] = w * x_train[i] + b

    # Plot fit
    plt.plot(x_train, predicted, c="b")
    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title("Profits vs. Population per city")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

    # Predict for populations
    predict1 = 3.5 * w + b
    print(f'For population = 35,000, we predict a profit of ${predict1*10000:.2f}')
    predict2 = 7.0 * w + b
    print(f'For population = 70,000, we predict a profit of ${predict2*10000:.2f}')