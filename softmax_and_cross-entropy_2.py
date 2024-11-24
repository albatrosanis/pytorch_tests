import torch
import torch.nn as nn
import numpy as np

#def cross_entropy(actual, predicted):
#    loss = -np.sum(actual * np.log(predicted))
#    return loss # / float(predicted.shape[0])

#y must be one hot encoded
#if class 0: [1 0 0]
#if class 1: [0 1 0]
#if class 2: [0 0 1]

#y = np.array([1, 0, 0])

#y_prediction has probabilities
#y_pred_good = np.array([0.7,0.2,0.1])
#y_pred_bad = np.array([0.1,0.3,0.6])
#l1 = cross_entropy(y , y_pred_good)
#l2 = cross_entropy(y , y_pred_bad)

#print(f'Loss1 numpy: {l1:.4f}')
#print(f'Loss2 numpy: {l2:.4f}')

#we can do this on pytorch

loss = nn.CrossEntropyLoss()
#3 samples for example
y = torch.tensor([2,0,1])

#nsampels * nclasses = 3*3
y_prediction_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[2.0,3.0,0.1]])
y_prediction_bad = torch.tensor([[2.5,1.0,0.1],[0.1,1.0,2.1],[0.0,3.0,0.1]])

l1 = loss(y_prediction_good, y)
l2 = loss(y_prediction_bad, y)

print(l1.item())
print(l2.item())

#to get actual prediction

_, prediciton1 = torch.max(y_prediction_good, 1)
_, prediciton2 = torch.max(y_prediction_bad, 1)

print(prediciton1)
print(prediciton2)

