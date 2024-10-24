#!/usr/bin/env python
# coding: utf-8

# # perceptron with linerally separable input 

import numpy as np


X = np.array([[0,0], [0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 1])

def activation(o):
    return 1 if o >= 0 else 0

def setp (Xshape):
    w = np.random.rand(Xshape)
    b = np.random.rand()
    return w, b

def prediction (w, b, i):
        linear_output = np.dot(i, w) + b
        return activation(linear_output)

def train_perceptron (X, y, alpha = 0.01, epochs = 10):
    Xshape = X.shape[1]
    w, b = setp(Xshape)
    
    for i in range(epochs):
        for i in range(len(X)):
            yhat = prediction(w, b, X[i])
            update = alpha * (y[i] - yhat)
            w += update * X[i]
            b += update
    return w, b

w, b = train_perceptron(X, y, alpha = 0.8, epochs = 50)
for i in range(len(X)):
    print(f"Input: {X[i]}, Prediction: {prediction(w, b, X[i])}")


# # question 11

# Consider a 2-dimensional data set in which all points with $x_1 > x_2$ belong to the positive class, and all points with $x_1 ≤ x_2$ belong to the negative class. Therefore, the true separator of the two classes is linear hyperplane (line) defined by $x_1 − x_2 = 0$. Now create a training data set with 20 points randomly generated inside the unit square in the positive quadrant. Label each point depending on whether or not the first coordinate $x_1$ is greater than its second coordinate $x_2$. Implement the perceptron algorithm, train it on the 20 points above, and test its accuracy on 1000 randomly generated points inside the unit square. Generate the test points using the same procedure as the training points.

# In[190]:


def generate_data(n_points):
    X = np.random.rand(n_points, 2)
    y = np.where(X[:, 0] > X[:, 1], 1, -1)
    return X, y

def activation(o):
    return 1 if o >= 0 else -1


def setp (Xshape):
    w = np.random.rand(Xshape)
    b = np.random.rand()
    return w, b


# In[192]:


def prediction (w, b, i):
        linear_output = np.dot(i, w) + b
        return activation(linear_output)


# In[193]:


def train_perceptron (X, y, alpha = 0.01, epochs = 10):
    Xshape = X.shape[1]
    w, b = setp(Xshape)
    
    for i in range(epochs):
        for i in range(len(X)):
            yhat = prediction(w, b, X[i])
            update = alpha * (y[i] - yhat)
            w += update * X[i]
            b += update
    return w, b


# In[194]:


def evaluate_perceptron(w, b, X, y):
    correct = 0
    for i in range(len(X)):
        yhat = prediction(w, b, X[i])
        if yhat == y[i]:
            correct += 1
    return correct / len(X)


# In[195]:


np.random.seed(42)
X_train, y_train = generate_data(20)
X_test, y_test = generate_data(1000)


# In[196]:


for i in range(len(X_train)):
    print(f"Input: {X_train[i]}, Prediction: {prediction(w, b, X_train[i])}")


# In[207]:


w, b = train_perceptron(X_train, y_train, alpha=0.12, epochs=25)


# In[208]:


accuracy = evaluate_perceptron(w, b, X_test, y_test)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")


# # chapter 2 question 10

# In[214]:


import numpy as np

# Define the function f(x) = sin(x) + cos(x)
def f(x):
    return np.sin(x) + np.cos(x)

# Define the derivative of f(x), which is f'(x) = cos(x) - sin(x)
def f_prime(x):
    return np.cos(x) - np.sin(x)


# In[218]:


# Define a function to compute f(f(f(f(x))))
def f_composed_4_times(x):
    f1 = f(x)
    f2 = f(f1)
    f3 = f(f2)
    f4 = f(f3)
    return f4


# In[219]:


# Compute the derivative of f(f(f(f(x)))) using the chain rule
def f_composed_4_times_derivative(x):
    f1 = f(x)
    f2 = f(f1)
    f3 = f(f2)
    f4 = f(f3)
    
    # Chain rule application for the derivative
    df4_dx = f_prime(f3) * f_prime(f2) * f_prime(f1) * f_prime(x)
    return df4_dx


# In[220]:


# Define the value of x = π/3
x_value = np.pi / 3

# Evaluate f(f(f(f(x)))) at x = π/3
f_value_at_x = f_composed_4_times(x_value)

# Evaluate the derivative of f(f(f(f(x)))) at x = π/3
f_prime_value_at_x = f_composed_4_times_derivative(x_value)

# Print the results
print("f(f(f(f(x)))) at x = π/3:", f_value_at_x)
print("Derivative of f(f(f(f(x)))) at x = π/3:", f_prime_value_at_x)


# In[221]:


import numpy as np

# Define the function f(x) = sin(x) + cos(x)
def f(x):
    return np.sin(x) + np.cos(x)

# Define the derivative of f(x), which is f'(x) = cos(x) - sin(x)
def f_prime(x):
    return np.cos(x) - np.sin(x)

# Recursive function to compute f composed n times
def f_composed_n_times(x, n):
    if n == 0:
        return x
    else:
        return f(f_composed_n_times(x, n-1))

# Recursive function to compute the derivative of f composed n times using the chain rule
def f_composed_n_times_derivative(x, n):
    if n == 0:
        return 1  # The derivative of x with respect to x is 1
    else:
        previous_f = f_composed_n_times(x, n-1)  # Compute f composed (n-1) times
        return f_prime(previous_f) * f_composed_n_times_derivative(x, n-1)

# Define the value of x = π/3
x_value = np.pi / 3

# Evaluate f(f(f(f(x)))) at x = π/3 using recursion
f_value_at_x = f_composed_n_times(x_value, 4)

# Evaluate the derivative of f(f(f(f(x)))) at x = π/3 using recursion
f_prime_value_at_x = f_composed_n_times_derivative(x_value, 4)

# Print the results
print("f(f(f(f(x)))) at x = π/3:", f_value_at_x)
print("Derivative of f(f(f(f(x)))) at x = π/3:", f_prime_value_at_x)


# In[222]:


import sympy as sp

# Define x as a symbolic variable
x = sp.symbols('x')

# Define the function f(x) = sin(x) + cos(x)
f_sympy = sp.sin(x) + sp.cos(x)

# Use sympy to compute f(f(f(f(x)))) recursively
f_composed_4_sympy = f_sympy
for _ in range(3):  # Already have 1 f, so we apply it 3 more times
    f_composed_4_sympy = f_sympy.subs(x, f_composed_4_sympy)

# Compute the derivative of f(f(f(f(x)))) using sympy
f_composed_4_derivative_sympy = sp.diff(f_composed_4_sympy, x)

# Evaluate both f(f(f(f(x)))) and its derivative at x = π/3
x_value = sp.pi / 3
f_value_at_x_sympy = f_composed_4_sympy.subs(x, x_value)
f_prime_value_at_x_sympy = f_composed_4_derivative_sympy.subs(x, x_value)

# Display the symbolic and evaluated results
print("Symbolic expression for f(f(f(f(x)))):")
sp.pprint(f_composed_4_sympy)

print("\nSymbolic expression for derivative of f(f(f(f(x)))):")
sp.pprint(f_composed_4_derivative_sympy)

print("\nValue of f(f(f(f(x)))) at x = π/3:", f_value_at_x_sympy)
print("Derivative of f(f(f(f(x)))) at x = π/3:", f_prime_value_at_x_sympy)

# Optionally, evaluate the results numerically
f_value_at_x_numeric = f_value_at_x_sympy.evalf()
f_prime_value_at_x_numeric = f_prime_value_at_x_sympy.evalf()

print("\nNumerical value of f(f(f(f(x)))) at x = π/3:", f_value_at_x_numeric)
print("Numerical value of derivative at x = π/3:", f_prime_value_at_x_numeric)

