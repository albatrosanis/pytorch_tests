#Use CIFAR-10 DATABASE (CAPTCHA) available in pytorch
#CNN = CONVENTIONAL NEURAL NETWORK : WORK ON IMAGE DATA APPLY THE CONVOLUTIONAL FILTER 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #run on gpu if here

#Hyper-parameters
num_epochs = 20 #so trqining not long = 2
batch_size = 4
learning_rate = 0.001

#dataset has PILIMAGE image of range [0, 1]
#we transform them to Tensor of normalized range [-1,1]

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform= transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform= transform, download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#implement conv net

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels (RGB), 6 output channels
        self.pool = nn.MaxPool2d(2, 2) # Max pooling with kernel size 2
        self.conv2 = nn.Conv2d(6, 16, 5) # Correctly define conv2
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Fully connected layer 1
        self.fc2 = nn.Linear(120, 84) # Fully connected layer 2
        self.fc3 = nn.Linear(84, 10) # Fully connected layer 3 (10 output classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # Use conv2 here
        x = x.view(-1, 16 * 5 * 5) # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation (handled in loss function)
        return x



model = Convnet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #origin shape: [4,3,32,32]= 4,3,1024
        #input layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch {epoch+1}/ {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

print('finish training !!!! connard')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labes = labels.to(device)
        outputs = outputs.to(device)
        #max returns(value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if(label == pred):
                n_class_correct[label] +=1
            n_class_samples[label] +=1
    acc = 100.0 * n_correct / n_samples

print(f'accuracy of the network = {acc} %')

for i in range(10):
    acc = 100.0*n_class_correct[i] / n_class_samples[i]
    print(f'accuracy of  {classes[i]} : {acc} %')
