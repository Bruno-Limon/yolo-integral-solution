# import requests
# from PIL import Image
# from io import BytesIO

# import torchvision.transforms as transforms
import torch
import time
import cv2
import os
from utils import *
from Detected_object import DetectedObject

from super_gradients.training import models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.processing import DetectionCenterPadding, StandardizeImage, ImagePermute, ComposeProcessing, DetectionLongestMaxSizeRescale


def detect_sg(model_name, frame):
    local_run = True
    if local_run == True:
        load_env_var()

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

    # model.set_dataset_processing_params(class_names=class_names,
    #                                     image_processor=image_processor,
    #                                     iou=0.35,
    #                                     conf=0.25)

    # image = cv2.imread('img2.PNG')
    # frame = cv2.resize(frame, (256, 256), interpolation= cv2.INTER_LINEAR)
    images_predictions = model.predict(frame, iou=.4, conf=.35, fuse_model=False)

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

# # Image can be both uploaded to colab server or by a direct URL
# image_path = "https://www.investorsinpeople.com/wp-content/uploads/2022/04/shutterstock_236097745.jpg"
# response = requests.get(image_path)
# image = Image.open(BytesIO(response.content))

# # preprocess
# import torchvision.transforms as transforms
# import torch
# # Prepare preprcoess transformations
# # We resize to [640, 640] by COCO's dataset default, which the model was pretrained on.
# preprocess = transforms.Compose([transforms.Resize([640, 640]),
#                                  transforms.PILToTensor()])

# # unsqueeze for [Batch x Channels x Width x Height] format
# transformed_image = preprocess(image).float().unsqueeze(0)

# # # load model
# model = models.get("yolox_n", pretrained_weights="coco")
# model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# model.eval()

# with torch.no_grad():
#   raw_predictions = model(transformed_image)

# predictions = YoloPostPredictionCallback(conf=0.1, iou=0.4)(raw_predictions)[0].numpy()
# bbox = predictions[:, 0:4]
# conf = predictions[:, 4]
# cls = predictions[:, 5]

# class Object(object):
#     pass
# detected_object = Object()

# detected_object.bbox = bbox
# detected_object.cls = cls
# detected_object.conf = conf

# print(detected_object.conf)


# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# plt.plot(bbox[:, [0, 2, 2, 0, 0]].T, bbox[:, [1, 1, 3, 3, 1]].T, '.-')
# plt.imshow(image.resize([640, 640]))
# plt.show()