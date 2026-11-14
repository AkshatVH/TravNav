import torch 
import torch.nn as nn

from train.vint_train.models.travnav.trav_modules import TraversabilityDecoder


model = TraversabilityDecoder()

dummy1 = torch.randn(1, 40, 12, 12)
dummy2 = torch.randn(1, 112, 6, 6)
dummy3 = torch.randn(1, 320, 3, 3)
target_grad = torch.randn(1, 1, 64, 64)


output = model(dummy1, dummy2, dummy3)
output.retain_grad()

# print("OUtput Shape: ", output.shape)
# print(torch.max(output))

criterion = torch.nn.MSELoss()
loss = criterion(output, target_grad)

loss.backward()

# print("Weight Grads: ", output.grad)
# print("Grad shape : ",output.grad.shape)
# print(torch.max(output.grad))
# print(torch.min(output.grad))
