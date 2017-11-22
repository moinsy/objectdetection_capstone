import os
from PIL import Image
import sys

if len(sys.argv) == 2:
    set = sys.argv[1]
else:
    set = 'train'

imagespath = '../data/{}/images'
resized_imagespath = '../data/{}/resized_images'

if not os.path.exists(resized_imagespath):
    os.makedirs(resized_imagespath)

images = os.listdir(imagespath)
for image in images:
    try:
        imagepath = os.path.join(imagespath,image)
        resized_imagepath = os.path.join(resized_imagespath,image)
        im = Image.open(imagepath)
        im.verify()

        im.thumbnail(1000,1000)
        im.save(resized_imagepath)

    except:
        pass