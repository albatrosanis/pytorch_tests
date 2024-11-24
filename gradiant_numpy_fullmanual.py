#how to optimize the model with auto grad computation using pytorch autorgad 1st implement linear regretion algo from scratch 1- Prediction:Manually
#2-Gradients Computation: Manually
#3-Loss Computation: Manually
#4-Parameter updates: Manually
#were gonna try to get to this
#Prediction: PyTorch Model
#Gradients Computation: Autograd
#Loss Computation: PyTorch Loss
#Parameter updates: PyTorch Optimizer

import numpy as np

# f = w * x

# Input and output data
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

# Initialize weight
w = 0.0

# Model prediction
def forward(x):
    return w * x

# Loss function: Mean Squared Error
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

# Gradient computation
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2 * x * (w*x - y)

def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y) / x.size

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 60

for epoch in range(n_iters):
    # Prediction (forward pass)
    y_pred = forward(X)

    # Loss computation
    l = loss(Y, y_pred)

    # Gradient computation
    dw = gradient(X, Y, y_pred)

    # Update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
