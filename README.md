Product Object Detection
-
This project aims at identifying a product object in a given image.

The project proposal and report are placed in the 'doc' directory

Below is a brief description of the files in this project,

- **prepData.py**, this file filters out what and how the data needs to be downloaded, for all the three sets(Train,Validation,Test).
The set to prepare can be passed as an argument.

- **dn_images.py**, this program downloads all the required files in 'data/{set}/images'. Set to be downaloaded is passed as an argument.

- **verifyandresize**, verifies and rescales the downloaded images to 'data/{set}/resized_images'

- **tfr_create/tfr_creator.py**, creates the tfrecord file for the tensorflow library to 'data' directory.

- **create_labels.py**, creates label_map to 'data/label_map.pbtxt' 

- **create_test_result_data.py**, creates a json file at 'data/test_images_result.json', which consists of predicted values of all the test images.

- **evaluate_mAP.py**, uses the json file to compare with actual ground truth values, calculates the Mean Average Precision (mAP)

- **object_detection.py**, runs the object detection using the frozen model, it takes IMAGE_URL or IMAGE_NAME, to pass IMAGE_NAME, test images needs to be downloaded, as the path directs to 'data/test/resized_images'.

- **object_detection_demo.ipynp**, notebook, which contains object detection demo with output. 

The training and evaluation was run through tensorflow object detection api, from its source folder, using commands present in 'data/commands' file.