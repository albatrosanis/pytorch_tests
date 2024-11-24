import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import math


class PokemonDataset(Dataset):
    def __init__(self):
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
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


# Usage example
dataset = PokemonDataset()
dataloader = DataLoader(dataset= dataset , batch_size=4, shuffle=True)

#u can do this :

#first_data = dataset[0]
#features, labels = first_data
#print(features, labels)

#use dataloader

#dataiter= iter(dataloader)
#data = next(dataiter)
#features, labels = data
#print(features, labels)

#we can iterate to not only get the next item like in th previous :
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples , n_iterations)

for epoch in  range(num_epochs):
    for i , (inputs, labels) in enumerate(dataloader):
        # forward backward , update to begin the training
        if(i+1) % 5 ==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

#torchvision.datasets.MNIST() LOAD DES DATASET DEJA in TORCH