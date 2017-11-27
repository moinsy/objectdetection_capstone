import tensorflow as tf
import numpy as np
from PIL import Image
import json
import label_map_util
import os

PATH_TO_GRAPH = 'data/frozen_model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'data/test/resized_images'

NUM_CLASSES = 17


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


images = os.listdir(PATH_TO_TEST_IMAGES_DIR)

test_image_results = {}

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for count, img in enumerate(images):
            try:
                image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, img)
                print ('Processing image {}: {}'.format(count, image_path))

                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                num = int(num[0])
                result = {'boxes': boxes[0][:num], 'scores': scores[0][:num], 'classes': classes[0][:num], 'num': num}
                test_image_results[img] = result
                print('results appended')
            except Exception as e:
                print (e)
                print ('Image not processed:{}'.format(image_path))
                pass



with open('data/test_images_result.json', 'w') as outfile:
    json.dump(test_image_results, outfile)