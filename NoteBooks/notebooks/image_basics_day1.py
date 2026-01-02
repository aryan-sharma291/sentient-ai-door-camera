import numpy as np

from PIL import Image

im = Image.open("picture.jpeg")



array = np.array(im)

print(array.shape)
print(array[0][0])   # prints the RGB value of the top-left pixel
im.show()

