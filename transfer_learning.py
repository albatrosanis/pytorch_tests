import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#here we used a trained model 

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #run on gpu if here

mean = np.array([0.485, 0.456,0.406])
std = np.array([0.229,0.224,0.225])

data_transforms ={
    'train':transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
}
#import data
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:0f}m {time_elapsed%60:.0f}s')
    print(f'Best val acc: {best_acc:4f}')

    #loab best mode weights
    model.load_state_dict(best_model_wts)
    return model

#we train the model a little bit with the new last layer
model = models.resnet18(pretrained = True)

#change last fully connected layer
num_features = model.fc.in_features

#createa a new layer and assing it to the last layer
model.fc = nn.Linear(num_features,2) #2 calasses bee and ants
model.to(device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)
#scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 7, gamma=0.1) #every 7 epochs lr * 0.1

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs= 10)

#************************************************************************************

#we train the model a little bit with the new last layer
model = models.resnet18(pretrained = True)
for param in model.parameters():
    param.requires_grad = False #frezz all the layer in the begining

#change last fully connected layer
num_features = model.fc.in_features

#createa a new layer and assing it to the last layer
model.fc = nn.Linear(num_features,2) #2 calasses bee and ants
model.to(device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)
#scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 7, gamma=0.1) #every 7 epochs lr * 0.1

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs= 10)

