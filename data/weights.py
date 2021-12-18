from numpy.lib.npyio import save
import torch
import os
import numpy as np
import sys

if __name__ == "__main__":
  model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
  save_param = open(sys.argv[1], "w")
  state_dict = model.state_dict()
  for name in state_dict:
    print(name, state_dict[name].shape)
    if name.endswith("num_batches_tracked"):
      continue
    tmp_data = state_dict[name].flatten().detach().numpy().astype(np.float32)
    for idx in range(tmp_data.__len__()):
      save_param.write(str(tmp_data[idx])+" ")
    save_param.write("\n")
  save_param.close()
