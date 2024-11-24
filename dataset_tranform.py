import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import math

#-----------------------------------------------------
#dataset = torchvision.datasets.MNIST(root='./pokemon', transform=torchvision.transforms.ToTensor())    #transform to tensor there is

#-----------------------------------------------------

class PokemonDataset(Dataset):
    def __init__(self, transform=None):  # Add transform as an optional parameter
        self.transform = transform  # Store the transform function
        
        # Load the CSV data
        xy = np.genfromtxt('./pokemon.csv', delimiter=",", dtype=None, encoding="utf-8", skip_header=1)
        
        # Ensure the data is read as a structured array
        self.data = [list(row) for row in xy]

        # Extract numeric columns (4th to 13th: Total, HP, Attack, Defense, etc.)
        self.x = torch.tensor([[float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12] == "True")] for row in self.data],dtype=torch.float32,)

        # Use the first column (`#`) as labels
        self.y = torch.tensor([int(row[0]) for row in self.data], dtype=torch.int64)
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class Totensor:
    def __call__(self, sample):
        #inputs, targets = sample
        #return torch.from_numpy(inputs), torch.from_numpy(targets)
        # Inputs and targets are already tensors, so just return them
        return sample
    
class MulTranform:
    def __init__(self , factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs ,target = sample
        inputs *= self.factor
        return inputs , target
  

# Instantiate the dataset with the transform
dataset = PokemonDataset(transform=Totensor()) #if transform = None type == np array not tensor
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
#Instantiate the dataset with mutilple  transforms  MulTranform and Totensor
composed = torchvision.transforms.Compose([Totensor(), MulTranform(4)])
dataset = PokemonDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
