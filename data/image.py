from os import close
from PIL import Image
import numpy as np
import sys
from torchvision.transforms import ToTensor

if __name__ == "__main__":
  image_path = sys.argv[1]
  output_path = sys.argv[2]
  img = Image.open(image_path).resize((224, 224))
  img = ToTensor()(img)
  img = img.numpy().flatten().astype(np.float32)
  buf = img.tobytes()
  file = open(output_path, "wb")
  file.write(buf)
  file.close()
