import numpy as np

data = [(1, 3), (2,5)]
m = -1
b = 5

def get_rmse(data, m, b):
    """Calculates Mean Squared Error"""
    n = len(data)
    squared_error = 0
    for x, y in data:
        # Find predicted y
        y_hat = m * x + b
        # Square difference between prediction and true value
        squared_error += (y - y_hat) ** 2
    # Get average squared difference
    mse = squared_error / n
    # Square root for original units
    return mse ** 0.5

print (get_rmse(data, m, b))

# activation functions

def linear(w, x, b):
    y_hat = w * x + b
    return y_hat

def releu(w, x, b):
    linear = w * x + b
    y_hat = linear * (linear > 0)
    return y_hat

def sigmoid(w, x, b):
    linear = w * x + b
    inf_to_zero = np.exp(-1 * linear)
    y_hat = 1 / (1 + inf_to_zero)
    return y_hat

w = 1
b = 0
x = [-10, -5, 0, 5, 10]

print('Linear:', [linear(w, i, b) for i in x])
print('ReLEU:', [releu(w, i, b) for i in x])
print('Sigmoid:', [sigmoid(w, i, b) for i in x])

def cross_entropy(y_hat, y_actual):
    """Infinite error for misplaced confidence"""
    loss = np.log(y_hat) if y_actual == 1 else np.log(1 - y_hat)
    return -1 * loss

y_hat = 0.9
y_actual = 1
print(cross_entropy(y_hat, y_actual))
y_hat = 0.1
y_actual = 1
print(cross_entropy(y_hat, y_actual))
y_hat = 0.95
y_actual = 1
print(cross_entropy(y_hat, y_actual))