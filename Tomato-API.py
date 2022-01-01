# import flask
import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from flask import Flask, request, Response, jsonify
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

app = Flask(__name__)

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
SAVE_IMAGES = False
IMAGES_NAME = "Detection"
IMAGES_COUNTER = 0

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-9')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'tomato.a12109dd-46a2-11ec-acc4-84c5a6ef94c0.jpg')

img = cv2.imread(IMAGE_PATH)
start_image = np.array(img)

input_tensor_start = tf.convert_to_tensor(np.expand_dims(start_image, 0), dtype=tf.float32)
start_detections = detect_fn(input_tensor_start)

# Start API:
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    global IMAGES_COUNTER

    data = request.json
    image_np = np.array(data['Image'])

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1

    # If wanted to save the image:
    if SAVE_IMAGES:
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                            detections['detection_boxes'],
                                                            detections['detection_classes']+label_id_offset,
                                                            detections['detection_scores'],
                                                            category_index,
                                                            use_normalized_coordinates=True,
                                                            max_boxes_to_draw=5,
                                                            min_score_thresh=.85,
                                                            agnostic_mode=False)

        plt.imshow(image_np_with_detections)
        plt.savefig("Results/%s - %d.jpg" % (IMAGES_NAME, IMAGES_COUNTER))
        IMAGES_COUNTER += 1

    # Response
    if detections['detection_scores'][0] >= 0.85:
        print("Tomato!")
        rsp = "Tomato"
    else:
        print("Not a Tomato")
        rsp = "Not a Tomato"

    return Response(rsp, status=200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)