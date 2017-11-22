import tensorflow as tf
import os
import dataset_util
from PIL import Image
import numpy as np
import base64
import cStringIO
import json

flags = tf.app.flags
flags.DEFINE_string('output_path', '/home/justdial/ImageSearch/tf_files/tf_od/oi_train.record', 'Path to output TFRecord')
flags.DEFINE_string('image_path', '/home/justdial/ImageSearch/data/oi_ml', 'Path to images')
flags.DEFINE_string('label_map','/home/justdial/ImageSearch/tf_files/tf_od/oi.pbtxt','Path to label map')
flags.DEFINE_string('label_json','/home/justdial/ImageSearch/scripts/tf_od_oi/label.json','Path to label json file')
FLAGS = flags.FLAGS


def create_tf(imagename, imagelabel):

    imageid = imagename.split('.')[0]

    filepath = os.path.join(FLAGS.image_path,imagename)
    image = Image.open(filepath)
    width, height = image.size


    filename = imagename
    image_format = image.format

    # buffer = cStringIO.StringIO()
    # image.save(buffer, format=image_format)
    # encoded_image_data = base64.b64encode(buffer.getvalue())

    encoded_image_data = tf.gfile.FastGFile(filepath, 'rb').read()
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for label in imagelabel[imageid].keys():
        labeldict = imagelabel[imageid][label]
        xmins.append(labeldict['xmin'])
        xmaxs.append(labeldict['xmax'])
        ymins.append(labeldict['ymin'])
        ymaxs.append(labeldict['ymax'])
        classes_text.append(str(labeldict['name']))
        classes.append(labeldict['number'])

    print ("Creating a record for {} ".format(filepath))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_image_data),
          'image/format': dataset_util.bytes_feature(image_format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # label_file = open(FLAGS.label_file,'a')


  # TODO(user): Write code to read in your dataset to examples variable
    images = os.listdir('/home/justdial/ImageSearch/data/oi_ml')

    with open(FLAGS.label_json,'r') as jsonf:
        image_lbl = json.load(jsonf)

    for image in images:
        imageid = image.split('.')[0]
        if imageid in image_lbl.keys():
            tf_example = create_tf(image, image_lbl)
            writer.write(tf_example.SerializeToString())
    writer.close()
    # label_file.close()

if __name__ == '__main__':
    tf.app.run()