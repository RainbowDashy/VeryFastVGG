from numpy.lib.npyio import save
import torch
import os
import numpy as np
import sys

if __name__ == "__main__":
  model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
  save_param = open(sys.argv[1], "w")
  for name, param in model.named_parameters():
    print(name, param.shape)
    tmp_data = param.flatten().detach().numpy().astype(np.float64)
    for idx in range(tmp_data.__len__()):
      save_param.write(str(tmp_data[idx])+" ")
    save_param.write("\n")
  save_param.close()
