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
num_epochs = 0 #so trqining not long = 2
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

def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#get rand training img
dataiter = iter(train_loader)
images , labels = next(dataiter)

#show img
imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6,16,5)
print(images.shape)
x = conv1(images)
print(x.shape)
x = pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)