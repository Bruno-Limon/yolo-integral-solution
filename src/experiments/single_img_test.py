from ultralytics import YOLO
import time
from classes.Detected_object import DetectedObject
from utils.detection_utils import set_initial_vars, generate_objects
from utils.postprocessing_utils import send_frame_info, aggregate_info, print_fps, send_agg_frame_info

from config import enable_local_work
enable_local_work()

init_vars = set_initial_vars()
# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model("C:\\Users\\Pavilion\\Desktop\\data\\x.PNG")  # return a list of Results objects

# Process results list
results_image = []
for r in results:
    results_image.append(r.cpu())

list_objects = generate_objects(DetectedObject, results_image, init_vars['labels_dict'], "ultralytics")
print(list_objects)
