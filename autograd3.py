import torch

weights = torch.ones(4, requires_grad=True) #requires_grad=True WHENEVER YOU CQLCULATE THE GRAD

optimizer =torch.optim.sgd(weights, lr=0.01)#STOCKASTIC GRADIANT DESCEND lr= learning rate
optimizer.step()
optimizer.zero_grad()#to empty the gradian again so that no acumulation

#resumer requires_grad=True THEN .BACKWARD FOR THE GRAD THEN .GRAD.ZERO()