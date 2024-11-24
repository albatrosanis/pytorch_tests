#calculate gradiant

import torch
x = torch.randn(3, requires_grad=False)
print(x)

y =x +2

print(y)

z =y*y*2
z =z.mean() #a mean opeareation to calculate gradinant th next

print(z)


z.backward() #dz/dx
print(x.grad)
#RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn