# -*- coding: utf-8 -*-
"""

@author: jens

RetinaNet detection model
"""
import tensorflow as tf
from pathlib import Path
import os

from flex_vision.bbox_detection.util import RetinaNet
from flex_vision.bbox_detection.util import DecodePredictions
from flex_vision.bbox_detection.util import get_backbone
from flex_vision.bbox_detection.util import prepare_image
from flex_vision.bbox_detection.util import visualize_detections
from flex_vision.bbox_detection.util import resize_image
from flex_vision.bbox_detection.util import save_bboxed_images


def get_detection_model():
    model_dir = "bbox_detection/retinanet_465_imgs/"

    path = Path(os.getcwd())
    pwd_model = os.path.join(path.parent, model_dir)

    num_classes = 1

    resnet50_backbone = get_backbone()
    model = RetinaNet(num_classes, resnet50_backbone)

    latest_checkpoint = tf.train.latest_checkpoint(pwd_model)
    
    model.load_weights(latest_checkpoint)

    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.8)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    return inference_model

def predict_truss(image, inference_model, pwd_detections=None, pwd_images=None, file_name=None):
    desired_size = 510

    # Resize image
    image_resized = resize_image(image, desired_size=desired_size)

    # Prediction
    img = tf.cast(image_resized, dtype=tf.float32)
    input_image, ratio = prepare_image(img)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    print(f'Prediction: {num_detections} detections')
    
    class_names = [int(x) for x in detections.nmsed_classes[0][:num_detections]]
    bboxes_pred = detections.nmsed_boxes[0][:num_detections] / ratio

    # Prediction visualisation
    visualize_detections(
        img,
        bboxes_pred,
        class_names,
        detections.nmsed_scores[0][:num_detections],
        save=True,
        pwd_detections=pwd_detections,
        file_name=file_name
    )

    bboxes = save_bboxed_images(image, bboxes_pred, desired_size, pwd_images=pwd_images, file_name=file_name)

    return num_detections, bboxes


