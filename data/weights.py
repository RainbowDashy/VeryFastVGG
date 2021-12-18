import torch
import numpy as np
import sys

if __name__ == "__main__":
  model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
  save_param = open(sys.argv[1], "wb")
  state_dict = model.state_dict()
  for name in state_dict:
    print(name, state_dict[name].shape)
    if name.endswith("num_batches_tracked"):
      continue
    tmp_data = state_dict[name].flatten().detach().numpy().astype(np.float32)
    buf = tmp_data.tobytes()
    save_param.write(buf)
  save_param.close()
