

from PIL import Image

im = Image.open("picture.jpeg")

box = (0,0 ,256 , 256)

region = im.crop(box)
region.show()

gray_image = Image.open("picture.jpeg")
gray_image.convert("L").show()

image_size = Image.open("picture.jpeg")
print("-----------------------------------------")
size = image_size.size
print(size)
print("-----------------------------------------")
new_size = (180, 390)
image_size = image_size.resize(new_size)
size = image_size.size
print(size)
image_size.show()