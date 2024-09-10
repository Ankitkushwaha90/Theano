Here is a comprehensive collection of Theano code examples, covering key functionalities like basic operations, matrix manipulations, gradients, shared variables, and more, to create a useful tutorial.

### 1. Basic Setup and Expressions
Theano uses symbolic expressions to define computations.

```python
import theano
import theano.tensor as T

# Define symbolic variables
x = T.dscalar('x')  # A scalar
y = T.dscalar('y')

# Define symbolic expression
z = x + y

# Compile the function
f = theano.function([x, y], z)

# Use the function
result = f(2, 3)
print("2 + 3 =", result)
```
### 2. Matrix Operations
You can easily define and manipulate matrices in Theano.

```python
import theano
import theano.tensor as T
import numpy as np

# Define symbolic matrices
A = T.dmatrix('A')
B = T.dmatrix('B')

# Define a matrix multiplication expression
C = T.dot(A, B)

# Compile the function
matrix_mul = theano.function([A, B], C)

# Test with NumPy arrays
A_val = np.array([[1, 2], [3, 4]])
B_val = np.array([[5, 6], [7, 8]])

result = matrix_mul(A_val, B_val)
print("Matrix multiplication result:\n", result)
```
### 3. Gradients (Automatic Differentiation)
Theano can automatically compute gradients of scalar expressions.

```python
import theano
import theano.tensor as T

# Define a symbolic variable
x = T.dscalar('x')

# Define a simple expression
y = x**2

# Compute the gradient of y with respect to x
grad_y = T.grad(y, x)

# Compile the gradient function
grad_func = theano.function([x], grad_y)

# Compute the gradient at x = 3
result = grad_func(3)
print("Gradient of x^2 at x=3:", result)
```
### 4. Working with Shared Variables
Shared variables are used to maintain state across multiple function calls.

```python
import theano
import theano.tensor as T

# Define a shared variable
state = theano.shared(0, name='state')

# Define an expression to update the state
new_state = state + 1

# Compile a function that updates the state
update_state = theano.function([], state, updates=[(state, new_state)])

# Print the initial state
print("Initial state:", state.get_value())

# Update the state multiple times
for _ in range(5):
    update_state()

# Print the updated state
print("Updated state:", state.get_value())

# Reset the state to 0
state.set_value(0)
print("Reset state:", state.get_value())
```
### 5. Logistic Regression Example
This example demonstrates a simple logistic regression using Theano.

```python
import theano
import theano.tensor as T
import numpy as np

# Input variables
X = T.dmatrix('X')  # Feature matrix
y = T.dvector('y')  # Labels

# Parameters to learn (weights and bias)
W = theano.shared(np.zeros((2,), dtype=theano.config.floatX), name='W')
b = theano.shared(0., name='b')

# Define the logistic function (sigmoid)
p_y_given_x = 1 / (1 + T.exp(-T.dot(X, W) - b))

# Define the prediction (0 or 1)
prediction = p_y_given_x > 0.5

# Define the loss function (log-likelihood)
loss = -T.mean(y * T.log(p_y_given_x) + (1 - y) * T.log(1 - p_y_given_x))

# Compute gradients
gW, gb = T.grad(loss, [W, b])

# Define the update rule (gradient descent)
updates = [(W, W - 0.1 * gW), (b, b - 0.1 * gb)]

# Compile the training function
train = theano.function([X, y], loss, updates=updates)

# Compile the prediction function
predict = theano.function([X], prediction)

# Generate some example data (AND gate)
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 0, 0, 1])

# Train for a few epochs
for epoch in range(1000):
    loss_val = train(X_data, y_data)

# Make predictions
predictions = predict(X_data)
print("Predictions after training:", predictions)
```
### 6. Softmax Function
Theano provides the softmax function, which is commonly used for multi-class classification.

```python
import theano
import theano.tensor as T
import numpy as np

# Define a symbolic matrix (logits)
x = T.dmatrix('x')

# Compute the softmax of each row
softmax_x = T.nnet.softmax(x)

# Compile the function
softmax_func = theano.function([x], softmax_x)

# Example logits
x_val = np.array([[1, 2, 3], [1, 5, 2]])

# Compute softmax
result = softmax_func(x_val)
print("Softmax result:\n", result)
```
### 7. Element-wise Operations
You can perform element-wise operations on tensors.

```python
import theano
import theano.tensor as T
import numpy as np

# Define symbolic vectors
a = T.dvector('a')
b = T.dvector('b')

# Element-wise multiplication
c = a * b

# Compile the function
elementwise_mul = theano.function([a, b], c)

# Test with NumPy arrays
a_val = np.array([1, 2, 3])
b_val = np.array([4, 5, 6])

result = elementwise_mul(a_val, b_val)
print("Element-wise multiplication result:", result)
```
### 8. Convolution
This example shows how to perform 2D convolutions in Theano.

```python
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
import numpy as np

# Define symbolic variables
input = T.tensor4(name='input')  # 4D tensor: (batch_size, channels, height, width)
filters = T.tensor4(name='filters')

# Perform 2D convolution
conv_output = conv2d(input, filters)

# Compile the function
conv2d_func = theano.function([input, filters], conv_output)

# Example data: batch of 1 image with 1 channel, and 1 filter
input_val = np.random.rand(1, 1, 5, 5).astype(theano.config.floatX)
filters_val = np.random.rand(1, 1, 3, 3).astype(theano.config.floatX)

# Perform the convolution
result = conv2d_func(input_val, filters_val)
print("Convolution result:\n", result)
```
### 9. RNN Example (Recurrent Neural Network)
Theano supports RNNs via scan functions.

```python
import theano
import theano.tensor as T
import numpy as np

# Define symbolic variables
x = T.dvector('x')  # Input sequence
h = T.dscalar('h')  # Hidden state at time t-1

# Define the recurrent relation
W_h = theano.shared(0.5)  # Hidden weight
W_x = theano.shared(0.8)  # Input weight

def recurrence(x_t, h_tm1):
    h_t = T.tanh(W_x * x_t + W_h * h_tm1)
    return h_t

# Use scan to apply the recurrence over the input sequence
h, _ = theano.scan(fn=recurrence, sequences=x, outputs_info=h)

# Compile the function
rnn = theano.function([x, h], h)

# Test with input data
x_val = np.array([1, 2, 3, 4, 5])
h_val = 0  # Initial hidden state
result = rnn(x_val, h_val)
print("RNN hidden states:\n", result)
```
### 10. Random Number Generation
Theano supports random number generation using shared random streams.

```python
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

# Create a random stream
rng = MRG_RandomStreams(seed=42)

# Define symbolic variables
rand_var = rng.uniform(low=0, high=1, size=(2, 2))

# Compile the function
rand_func = theano.function([], rand_var)

# Generate random numbers
result = rand_func()
print("Random numbers:\n", result)
```
This Theano tutorial includes examples of basic symbolic operations, matrix manipulations, gradients, shared variables, logistic regression, RNNs, and more. These examples demonstrate how Theano's symbolic computation framework can be used for machine learning, optimization, and deep learning tasks.
