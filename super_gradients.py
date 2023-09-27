# from super_gradients.training import models
# from super_gradients.common.object_names import Models

# model = models.get("yolox_n", pretrained_weights="coco")

# images_predictions = model.predict("src/img1.PNG", iou=0.5, conf=0.4)
# images_predictions.show()

import requests
from PIL import Image
from io import BytesIO

import torchvision.transforms as transforms
import torch

from super_gradients.training import models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback


# Image can be both uploaded to colab server or by a direct URL
image_path = "https://www.investorsinpeople.com/wp-content/uploads/2022/04/shutterstock_236097745.jpg"
response = requests.get(image_path)
image = Image.open(BytesIO(response.content))

# preprocess
import torchvision.transforms as transforms
import torch
# Prepare preprcoess transformations
# We resize to [640, 640] by COCO's dataset default, which the model was pretrained on.
preprocess = transforms.Compose([transforms.Resize([640, 640]),
                                 transforms.PILToTensor()])

# unsqueeze for [Batch x Channels x Width x Height] format
transformed_image = preprocess(image).float().unsqueeze(0)


# # load model
model = models.get("yolox_n", pretrained_weights="coco")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

with torch.no_grad():
  raw_predictions = model(transformed_image)

predictions = YoloPostPredictionCallback(conf=0.1, iou=0.4)(raw_predictions)[0].numpy()
bbox = predictions[:, 0:4]
conf = predictions[:, 4]
cls = predictions[:, 5]

class Object(object):
    pass
detected_object = Object()

detected_object.bbox = bbox
detected_object.cls = cls
detected_object.conf = conf

print(detected_object.conf)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.plot(bbox[:, [0, 2, 2, 0, 0]].T, bbox[:, [1, 1, 3, 3, 1]].T, '.-')
plt.imshow(image.resize([640, 640]))
plt.show()