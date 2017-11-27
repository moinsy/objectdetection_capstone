import tensorflow as tf
import numpy as np
import os
from PIL import Image
import urllib.request
import label_map_util
import visualization_utils as vis_util
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('PATH_TO_GRAPH', 'data/frozen_model/frozen_inference_graph.pb',
                    'Path to a frozen graph.')
flags.DEFINE_string('PATH_TO_LABELS', 'data/label_map.pbtxt',
                    'Path to label map file')
flags.DEFINE_string('PATH_TO_TEST_IMAGES_DIR', 'data/test/resized_images',
                    'Path to test set images')
flags.DEFINE_string('IMAGE_NAME', '',
                    'name of the image file in the test set')
flags.DEFINE_string('IMAGE_URL', '', 'url of the image to detect objects')

FLAGS = flags.FLAGS

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def down(url, path):
    f = open(path, 'wb')
    f.write(urllib.request.urlopen(url, timeout=5).read())
    f.close()

def objdetfromname(image_name):
    image_path =  os.path.join(FLAGS.PATH_TO_TEST_IMAGES_DIR, image_name)
    return objdet(image_path)

def objdetfromurl(image_url):
    format = image_url.split('.')[-1]
    dn_path = 'data/image/to_detect.'+format
    down(image_url,dn_path)

    return objdet(dn_path)

def get_label(id):
    pdts = pd.read_csv('data/products.csv')
    return pdts[pdts['id'] == id].labelname.values[0]


def objdet(image_path):

    PATH_TO_GRAPH = FLAGS.PATH_TO_GRAPH
    PATH_TO_LABELS = FLAGS.PATH_TO_LABELS
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


            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            format = image_path.split('.')[-1]
            save_path = 'data/image/after_detect.'+format
            img = Image.fromarray(image_np,'RGB')
            img.save(save_path)

            scores = [x for x in scores[0] if x > 0.5]

            boxes, classes, num = boxes[0][:len(scores)], classes[0][:len(scores)], len(scores)

            res = {}
            for i,clas in enumerate(classes):
                res[get_label(clas)] = {'bbox': boxes[i], 'score':scores[i]}

            return save_path,res


def main(_):
    if FLAGS.IMAGE_NAME:
        s_path,res = objdetfromname(FLAGS.IMAGE_NAME)
        Image.open(s_path).show()
        print (res)

    elif FLAGS.IMAGE_URL:
        s_path,res = objdetfromurl(FLAGS.IMAGE_URL)
        Image.open(s_path).show()
        print (res)
    else:
        print ('Please enter Image name or url')


if __name__ == '__main__':
    tf.app.run()