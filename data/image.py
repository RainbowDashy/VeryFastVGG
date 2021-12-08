from PIL import Image
import numpy as np
import sys

if __name__ == "__main__":
  image_path = sys.argv[1]
  output_path = sys.argv[2]
  resized_image = Image.open(image_path).resize((224, 224))
  img_data = np.asarray(resized_image).astype("float32")
  img_data = img_data.flatten()
  np.savetxt(output_path, img_data, newline=" ")
