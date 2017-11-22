import os
from PIL import Image
import sys

if len(sys.argv) == 2:
    set = sys.argv[1]
else:
    set = 'train'

imagespath = '../data/{}/images'.format(set)
resized_imagespath = '../data/{}/resized_images'.format(set)

if not os.path.exists(resized_imagespath):
    os.makedirs(resized_imagespath)

images = os.listdir(imagespath)
for image in images:
    try:
        imagepath = os.path.join(imagespath,image)
        resized_imagepath = os.path.join(resized_imagespath,image)
        im = Image.open(imagepath)
        im.verify()
        print ('Verifying image: {}'.format(imagepath))

        im = Image.open(imagepath)
        im.thumbnail((1000,1000))

        im.save(resized_imagepath)
        print ('Resized image to : {}'.format(resized_imagepath))
        print('Total images resized: {}'.format(len(os.listdir(resized_imagespath))))
    except Exception as e:
        print (e)
        print ('Image not Verified, hence not resized : {}'.format(imagepath))
        pass