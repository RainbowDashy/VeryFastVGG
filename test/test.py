import torchvision.models as models
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import sys

if __name__ == "__main__":
  input_path = sys.argv[1]
  output_path = sys.argv[2]
  vgg11_bn = models.vgg11_bn(pretrained=True)
  vgg11_bn.eval()
  img = Image.open(input_path).resize((224, 224))
  img = ToTensor()(img)
  batch = torch.unsqueeze(img, 0)
  out = vgg11_bn(batch)
  out = out.detach().numpy()
  np.savetxt(output_path, out, fmt="%.8f", delimiter="\n")
