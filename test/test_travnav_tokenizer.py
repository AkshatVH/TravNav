import torch
import torch.nn as nn 
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]  # Adjust as needed to reach the folder containing 'train'
sys.path.append(str(root_dir))

from train.vint_train.models.travnav.trav_modules import TraversabilityTokenizer 

model = TraversabilityTokenizer()

dummy = torch.randn(5, 1, 64, 64)
target_grad = torch.randn(5, 4, 256)

output = model(dummy)

output.retain_grad()

print("Output shape: ", output.shape)
print(torch.max(output))

criterion = torch.nn.MSELoss()
loss = criterion(output, target_grad )

loss.backward()

print("Weight Grads: ", output.grad)
print("Grad shape : ",output.grad.shape)
print(torch.max(output.grad))
print(torch.min(output.grad))
