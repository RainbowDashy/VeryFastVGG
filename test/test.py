import torchvision.models as models
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torch.autograd import Variable

vgg11_bn = models.vgg11_bn(pretrained=True)
img = Image.open("data/image.png").resize((224, 224))
# transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
# img_data = transform(img)
# vgg11_bn.eval()
img = ToTensor()(img)
# out = F.interpolate(img, (224, 224))  #The resize operation on tensor.
batch = torch.unsqueeze(img, 0)
out = vgg11_bn(batch)
print(out)