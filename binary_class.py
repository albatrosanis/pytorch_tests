import torch
import torch.nn as nn

#Multiclass problem
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet1).__init__()
        #SET UP Layers and activation functions
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu == nn.ReLU()
        self.linear2 = nn.Linear(hidden_size ,1) #output size = 1 always

    def forward(self , x):
        #apply layers
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #sigmoid at the end
        y_pred = torch.sigmoid(out) #implement sigmoid function
        return y_pred
    
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()