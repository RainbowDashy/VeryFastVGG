import torchvision.models as models
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

vgg11_bn = models.vgg11_bn(pretrained=True)
vgg11_bn.eval()
img = Image.open("data/image.png").resize((224, 224))
img = ToTensor()(img)
batch = torch.unsqueeze(img, 0)
out = vgg11_bn(batch)
print(out)