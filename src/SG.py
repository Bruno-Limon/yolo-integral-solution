from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLOX_N, pretrained_weights="coco")