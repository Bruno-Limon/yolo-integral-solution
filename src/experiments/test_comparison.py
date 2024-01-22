import os
import sys
import cv2
import time
import json

sys.path.append('../yolo-traffic-mobility')
from config import enable_local_work
enable_local_work()

sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils.postprocessing_utils import get_labels_dict
from utils.detection_utils import generate_objects
from classes.Detected_object import DetectedObject

sys.path.append(os.path.join(os.getcwd(), 'src', 'YOLOX'))
from YOLOX.yolox.exp import get_exp
from YOLOX.tools.detect import process_frame
from ultralytics import YOLO


labels_dict = get_labels_dict()
path = "C:\\Users\\Pavilion\\Desktop\\data\\1.jpg"
list_exp = ["yolox-nano", "yolox-s"]
list_sizes = ["yolox_nano", "yolox_s"]
list_sizes_v8 = ["n", "s"]
list_resolutions = ["640,640"]
list_confidence = [.15, .30]

list_bbox_yolox = []
list_bbox_yolov8 = []

"""
YOLOX
"""
i = 0
dict_times_yolox = {}
for exp_name, size in zip(list_exp, list_sizes):
    model_yolox = get_exp(exp_file=None, exp_name=exp_name)
    for resolution in list_resolutions:
        for confidence in list_confidence:
            frame = cv2.imread(path)
            start = time.time()
            results_image, img_info = process_frame(model_name=f"src/models/{size}.pth", exp=model_yolox, frame=frame,
                                                    conf=confidence, iou=".45", input_size=resolution)

            list_objects = generate_objects(DetectedObject, results_image, labels_dict, "yolox")
            end = time.time()
            infer_time = end-start

            [obj.draw_boxes(frame, labels_dict) for obj in list_objects]
            i += 1
            print(f"{i}-{size}_{resolution}_{confidence}")
            res = str(resolution.split(',')[0])
            conf = str(confidence)
            conf = conf.replace(".", "")

            dict_times_yolox[f"{i}-{size}_{res}_{conf}"] = infer_time
            cv2.imwrite(f"experiments/img_results/{i}-{size}_{res}_{conf}.jpg",  frame)

            for cls, bbox, score in zip(results_image.cls, results_image.bbox, results_image.conf):
                if cls in [0,1,2,3,5,7]:
                    obj_dict = {"category_id": cls,
                                "bbox": bbox,
                                "file_name": f"{i}-{size}_{res}_{conf}",
                                "score": score}
                    list_bbox_yolox.append(obj_dict)

json_bbox_yolox = json.dumps(list_bbox_yolox, indent=4)
with open("experiments/test_results/bbox_yolox.json", "w") as outfile:
    outfile.write(json_bbox_yolox)

json_dict_times = json.dumps(dict_times_yolox, indent=4)

with open("experiments/test_results/dict_times.json", "w") as outfile:
    outfile.write(json_dict_times)

"""
YOLOv8
"""
# i = 0
# dict_times_yolov8 = {}
# for size in list_sizes_v8:
#     model_ultralytics = YOLO(f"yolov8{size}.pt")
#     for resolution in list_resolutions:
#         for confidence in list_confidence:
#             frame = cv2.imread(path)
#             res = int(resolution.split(',')[0])
#             conf = str(confidence)
#             conf = conf.replace(".", "")

#             start = time.time()
#             results_ultralytics = model_ultralytics(frame, save=False, stream=True, verbose=False, conf=confidence,
#                                                     iou=0.4, classes=[0,1,2,3,5,7], imgsz=res)
#             i += 1

#             results_image = []
#             for r in results_ultralytics:
#                 results_image.append(r.cpu())
#                 boxes = r.boxes.numpy()
#                 for box in boxes:
#                     obj_dict = {"category_id": int(box.cls),
#                                 "bbox": box.xyxy[0].tolist(),
#                                 "file_name": f"{i}-yolov8{size}_{str(res)}_{conf}",
#                                 "score": float(box.conf)}
#                     list_bbox_yolov8.append(obj_dict)

#             list_objects = generate_objects(DetectedObject, results_image, labels_dict, "ultralytics")
#             end = time.time()
#             infer_time = end-start

#             [obj.draw_boxes(frame, labels_dict) for obj in list_objects]
#             print(f"{i}-yolov8{size}_{resolution}_{confidence}")

#             dict_times_yolov8[f"{i}-yolov8{size}_{str(res)}_{conf}"] = infer_time
#             cv2.imwrite(f"src/experiments/test_results/{i}-yolov8{size}_{str(res)}_{conf}.jpg", frame)

# json_bbox_yolov8 = json.dumps(list_bbox_yolov8, indent=4)
# with open("src/experiments/test_results/bbox_yolov8.json", "w") as outfile:
#     outfile.write(json_bbox_yolov8)

# dict_times_yolox.update(dict_times_yolov8)
json_dict_times = json.dumps(dict_times_yolox, indent=4)

with open("experiments/test_results/dict_times.json", "w") as outfile:
    outfile.write(json_dict_times)