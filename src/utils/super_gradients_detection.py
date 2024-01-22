# import requests
# from PIL import Image
# from io import BytesIO

# import torchvision.transforms as transforms
import torch
import time
import cv2
from utils.postprocessing_utils import *
from utils.detection_utils import generate_objects
import config
from src.classes.Detected_object import DetectedObject

from super_gradients.training import models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.processing import DetectionCenterPadding, StandardizeImage, ImagePermute, ComposeProcessing, DetectionLongestMaxSizeRescale


def detect_sg(model_name, frame, conf, iou):
    if config.env_vars_local == True:
        from dotenv import load_dotenv
        load_dotenv()

    start = time.time()

    class_names = ["person",
                   "bicycle",
                   "car"]

    image_processor = ComposeProcessing(
        [
            DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
            DetectionCenterPadding(output_shape=(640, 640), pad_value=114),
            StandardizeImage(max_value=255.0),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    model = models.get(model_name, pretrained_weights="coco")
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    images_predictions = model.predict(frame, iou=float(iou), conf=float(conf), fuse_model=False)

    class Object(object):
        pass
    detected_object = Object()

    for image_prediction in images_predictions:
        class_names = image_prediction.class_names
        labels = image_prediction.prediction.labels
        confidence = image_prediction.prediction.confidence
        bboxes = image_prediction.prediction.bboxes_xyxy

        detected_object.class_names = class_names
        detected_object.bbox = bboxes.tolist()
        detected_object.cls = labels.astype(int).tolist()
        detected_object.conf = confidence.tolist()

    end = time.time()
    elapsed_time = end-start

    return detected_object, elapsed_time

if __name__ == "__main__":
    path = "C:\\Users\\Pavilion\\Desktop\\4.jpg"
    frame = cv2.imread(path)
    labels_dict = get_labels_dict()

    results_image, infer_time = detect_sg("yolox_n", frame)
    list_objects = generate_objects(DetectedObject, results_image, labels_dict)

    [obj.draw_boxes(frame, labels_dict) for obj in list_objects]
    for obj in list_objects:
        print(obj.label_str)

    cv2.imshow('Demo', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
