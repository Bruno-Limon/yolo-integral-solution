import os
import sys
import cv2
import json

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'YOLOX'))
from YOLOX.yolox.exp import get_exp
from YOLOX.tools.detect import process_frame


image_dir = os.listdir("src/experiments/subval2017/test")
list_files_images = [f"src/experiments/subval2017/test/{file_name}" for file_name in image_dir]
list_images = [cv2.imread(file) for file in list_files_images]

list_objects = []
model_yolox = get_exp(exp_file=None, exp_name="yolox-nano")
for i, (img, file_name) in enumerate(zip(list_images, image_dir)):
    results_image, img_info = process_frame(model_name="src/models/yolox_nano.pth", exp=model_yolox, frame=img,
                                            conf=.1, input_size="640,640")

    for cls, bbox, score in zip(results_image.cls, results_image.bbox, results_image.conf):
        if cls in [0,1,2,3,5,7]:
            obj_dict = {"category_id": cls,
                        "bbox": bbox,
                        "file_name": file_name,
                        "score": score}
            list_objects.append(obj_dict)
    print(f"Image number: {i+1}")

json_objects = json.dumps(list_objects, indent=4)
with open("src/experiments/detections_nano.json", "w") as outfile:
    outfile.write(json_objects)

