import torch 
from efficientnet_pytorch import EfficientNet


net = EfficientNet.from_name("efficientnet-b0", in_channels=3)
x = torch.randn(2, 3, 96, 96)
for name, feat in net.extract_endpoints(x).items():
    print(name, tuple(feat.shape))