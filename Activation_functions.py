#it become non linear we use activation functions

import torch 
import torch.nn as nn
import torch.nn.functional as F

#option1 (crate nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet).__init__()
        #define layere
        self.linear1 = nn.Linear(input_size,hidden_size) #lienar layer
        self.relu == nn.ReLU() #relu activaiton function contain all activation funvtion
        self.linear2 = nn.Linear(hidden_size ,1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self , x):
        #call all the previoius functions
        out = self.linear1(x)
        out = self.relu(out)
        #WE CAN HAVE 
        #NN.SIGMOID
        #NN.SOFTMAX
        #NN.TANH
        #NN.LEAKYRELU
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
#option2 (use activation functiuons directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet).__init__()
        #only define linearr layer
        self.linear1 = nn.Linear(input_size,hidden_size) #lienar layer
        self.linear2 = nn.Linear(hidden_size ,1)
    #apply finctions
    def forward(self , x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        #ALSO:
        #TORCH.TANH
        #TORCH.SOFTMAX .... TORCH.NN.FUNCTIONEAL = F.TANH.....
        return out

#BOTH OPTIONS DO THE SAME THING