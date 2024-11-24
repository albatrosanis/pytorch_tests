import numpy as np


#this is an example -----------------------------------------------------------------------------------
#data = np.loadtxt('pokemon.csv')
#training loop
#for epoch in range(1000):
#    x , y = data
#    #foward + backward + weight updates.......
#----------------------------------------------------------------------------------------------------- time consuming if we do gradiant calculation on the whole data we divide the sample so that oit look like this

#training loop
#for epoch in range(1000):
#    #loop over all batches
#    for i in range(total_batches):
#        x_batch, y_batches = ......
#
#use DataSet and DataLoader to load wine.csv

#first there are some terms suivant

#explains the concepts of epochs, batch size, and iterations in the context of training machine learning models. Here’s a breakdown:

#Epoch: A single epoch is defined as one complete forward and backward pass through all the training samples in the dataset.

#Batch Size: The number of training samples processed in a single forward and backward pass.

#Iterations: The number of batches required to complete one epoch. It is calculated as:

#Iterations= Total Training Samples / Batch Size
 
#For example, if there are 100 training samples and the batch size is 20, then:
#Iterations per Epoch=100/20=5
