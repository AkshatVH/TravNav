import torch 
import torch.nn as nn

from trian.vit_tain.models.travnav.trav_modules import TraversabilityDecoder


model = TraversabilityDecoder()

dummy1 = torch.randn(3, 40, 12, 12)
dummy2 = torch.randn(3, 112, 6, 6)
dummy3 = torch.randn(3, 320, 3, 3)

output = model(dummy1, dummy2, dummy3)

print("OUtput Shape: ", output.shape)

