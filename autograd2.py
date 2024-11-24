import torch

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):#the more wee add iteration the more vercote are acu;ulated 3333 then 6666 9999 ... ....etc 
    model_output = (weights*3).sum() #a dummy opertation that will simulate the ouytput of a model

    model_output.backward() #give the gradiant

    print(weights.grad) 

    weights.grad.zero_() #to empty the gradian again so that its 3333 then 3333 no acumulation