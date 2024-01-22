from collections import defaultdict
import os
import gc
import sys
import torch
import numpy as np
import cv2
import time
from src.classes.Detected_object import DetectedObject
from src.utils.postprocessing_utils import get_labels_dict, get_zone_poly, get_bbox_xywh, draw_keypoints
from src.utils.postprocessing_utils import get_door_polygons, count_objs, count_zone


# yolo_x through super-gradients
if os.environ['LIBRARY'] == "supergradients":
    from utils.super_gradients_detection import detect_sg

# yolo_x from source
if os.environ['LIBRARY'] == "yolox":
    sys.path.append(os.path.join(os.getcwd(), 'src', 'YOLOX'))
    from YOLOX.yolox.exp import get_exp
    from YOLOX.tools.detect import process_frame

# yolov8 through ultralytics
if os.environ['LIBRARY'] == "ultralytics":
    from ultralytics import YOLO

# global flag to evaluate if environment variables are True
ENV_VAR_TRUE_LABEL = "true"

# list of classes to detect
list_classes = [int(x) for x in os.environ['DETECTED_CLASSES'].split(",")]

# enable rtsp capture for opencv
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENV_VAR_TRUE_LABEL = "true"

def set_initial_vars()->dict:
    """
    setting up necessary variables that need to be declared before the detection loop
    starts on each frame, so they are initialized at the beginning
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    init_vars = {}
    # dictionary to store processing times
    init_vars['times_dict'] = {}
    # dictionary to map the class number obtained with yolo with its name and color for bbox
    init_vars['labels_dict'] = get_labels_dict()
    # zone to count people in
    init_vars['zone_poly'] = get_zone_poly(os.environ['ZONE_COORDS'])
    # crossing zone to calculate speed
    # init_vars['cross_poly'] = get_zone_poly(os.environ['CROSS_COORDS'])

    # dictionaries containing id of people entering/leaving and sets to count them
    init_vars['people_entering_dict'] = {}
    init_vars['entering'] = set()
    init_vars['people_leaving_dict'] = {}
    init_vars['leaving'] = set()

    # store the track history
    init_vars['track_history_dict'] = defaultdict(lambda: [])
    # store the amount of frames spent in scene
    init_vars['time_in_scene_dict'] = defaultdict(int)
    # store the amount of frames spent inside zone
    init_vars['time_in_zone_dict'] = defaultdict(int)
    # store the amount of frames spent inside crossing zone
    # init_vars['time_in_cross_dict'] = defaultdict(int)

    # initialize frame skipping mechanism
    init_vars['is_stream'] = os.environ['IS_STREAM'] == ENV_VAR_TRUE_LABEL
    init_vars['frames_to_skip'] = 5

    # initialize empty lists for calculating time for aggregated messages
    init_vars['list_times'] = []
    init_vars['list_aggregates'] = [[],[],[],[],[],
                                    [],[],[],[]]
    init_vars['time_interval_counter'] = 1

    # initialize imread and masking for alerts logos
    init_vars['logo_size'] = 50
    init_vars['logo_toomanypeople'] = cv2.imread('src/alert-logo/toomanypeople.png')
    init_vars['logo_toomanypeople'] = cv2.resize(init_vars['logo_toomanypeople'], (init_vars['logo_size'], init_vars['logo_size']))

    init_vars['logo_mandown'] = cv2.imread('src/alert-logo/mandown.png')
    init_vars['logo_mandown'] = cv2.resize(init_vars['logo_mandown'], (init_vars['logo_size'], init_vars['logo_size']))

    # create mask
    init_vars['img2gray_toomanypeople'] = cv2.cvtColor(init_vars['logo_toomanypeople'], cv2.COLOR_BGR2GRAY)
    init_vars['ret_toomanypeople'], init_vars['mask_toomanypeople'] = cv2.threshold(init_vars['img2gray_toomanypeople'],
                                                                                    1, 255,
                                                                                    cv2.THRESH_BINARY)

    init_vars['img2gray_mandown'] = cv2.cvtColor(init_vars['logo_mandown'], cv2.COLOR_BGR2GRAY)
    init_vars['ret_mandown'], init_vars['mask_mandown'] = cv2.threshold(init_vars['img2gray_mandown'],
                                                                        1, 255,
                                                                        cv2.THRESH_BINARY)
    return init_vars

def connect_video_source()->tuple:
    """
    getting video capture for each frame
    """
    t0 = time.time()
    cap = cv2.VideoCapture(os.environ['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    read_frame_time = (time.time() - t0)


    return cap, width, height, fps, read_frame_time

def load_model_size(model_size:str, library:str, do_pose:str)->str:
    """
    given the size and library of models to use, returns the correct way to
    initialize their model

    params:
        model_size: size of model to use, either nano, small or medium
        library: library to use either yolox or yolov8
        do_pose: wether pose_detection model is used
    """
    model_detection = None
    model_pose = None
    list_sizes = ["nano", "small", "medium", "large"]

    if model_size in list_sizes:
        for size in list_sizes:
            if library == "supergradients" and model_size == size:
                model_detection = f"yolox_{size[0]}"

            if library == "yolox" and model_size == size:
                model_detection = f"src/models/yolox_{size[0]}.pth"

            if library == "ultralytics" and model_size == size:
                model_detection = f"yolov8{size[0]}.pt"

            if library == "ultralytics" and model_size == size and do_pose == ENV_VAR_TRUE_LABEL:
                model_pose = f"yolov8{size[0]}-pose.pt"
    else:
        raise ValueError(f"Error in '{model_size}': 'model_size' must be one of {list_sizes}")

    return model_detection, model_pose

def load_model():
    """
    loads model depending on library and model size to use
    """
    list_libraries = ["ultralytics", "yolox", "supergradients"]

    if os.environ['LIBRARY'] not in list_libraries:
        raise ValueError(f"Error in '{os.environ['LIBRARY']}': 'library' must be one of {list_libraries}")

    if os.environ['LIBRARY'] == "ultralytics":
        model_v8_pose = None
        detection_size, pose_size = load_model_size(os.environ['MODEL_SIZE'],
                                                    os.environ['LIBRARY'],
                                                    os.environ['DO_POSE'])

        model_v8_detection = YOLO(detection_size) # tracking and object detection
        if os.environ['DO_POSE'] == ENV_VAR_TRUE_LABEL:
            model_v8_pose = YOLO(pose_size) # pose detection
        return model_v8_detection, model_v8_pose

    if os.environ['LIBRARY'] == "yolox":
        if os.environ['MODEL_SIZE'] in ["nano", "tiny"]:
            model_yolox = get_exp(exp_file=None, exp_name=f"yolox-{os.environ['MODEL_SIZE']}")
            return model_yolox, None
        else:
            model_yolox = get_exp(exp_file=None, exp_name=f"yolox-{os.environ['MODEL_SIZE'][0]}")
            return model_yolox, None
    return None

def compute_detection(model_detection, model_pose, frame, labels_dict:dict)->tuple:
    """
    depending on the library used, inference is done on each frame and a list of detected objects
    of class DetectedObject is created and returned together with the inference time

    params
        model_detection: model file to use for obj detection, obtained with the func load_model(s)
        model_pose: model to use for pose detection
        frame: current frame given by opencv
        labels_dict: dictionary with corresponding labels for categories of objects
    """
    list_res_pose = None
    if os.environ['LIBRARY'] == "yolox":
        detection_size, pose_size = load_model_size(os.environ['MODEL_SIZE'],
                                                    os.environ['LIBRARY'],
                                                    os.environ['DO_POSE'])
        results_image, img_info, infer_time_yx = process_frame(model_name=detection_size,
                                                            exp=model_detection,
                                                            frame=frame,
                                                            conf=os.environ['CONFIDENCE'],
                                                            iou=os.environ['IOU'],
                                                            input_size=os.environ['INPUT_SIZE'])
        list_objects = generate_objects(DetectedObject, results_image,
                                        labels_dict, os.environ['LIBRARY'])
        infer_time = infer_time_yx

    if os.environ['LIBRARY'] == "supergradients":
        detection_size, pose_size = load_model_size(os.environ['MODEL_SIZE'],
                                                    os.environ['LIBRARY'],
                                                    os.environ['DO_POSE'])
        results_image, infer_time = detect_sg(detection_size, frame,
                                              os.environ['CONFIDENCE'], os.environ['IOU'])
        list_objects = generate_objects(DetectedObject, results_image,
                                        labels_dict, os.environ['LIBRARY'])

    if os.environ['LIBRARY'] == "ultralytics":
        input_size = int(os.environ['INPUT_SIZE'].split(',')[0])

        if os.environ['DO_POSE'] == ENV_VAR_TRUE_LABEL:
            results_pose = model_pose(frame,
                                      save=False,
                                      stream=True,
                                      verbose=False,
                                      conf=float(os.environ['CONFIDENCE']),
                                      iou=float(os.environ['IOU']),
                                      imgsz=input_size,
                                      classes=list_classes)

            # processing pose results
            list_res_pose = []
            list_time_pose = []
            for r in results_pose:
                list_res_pose.append(r.cpu())
                list_time_pose.append(r.speed)

            time_pose = (list_time_pose[0]['preprocess'] +
                         list_time_pose[0]['inference'] +
                         list_time_pose[0]['postprocess'])
            time_pose /= 1000

        if os.environ['DO_TRACKING'] == ENV_VAR_TRUE_LABEL:
            results_detection = model_detection.track(frame,
                                                      save=False,
                                                      stream=True,
                                                      verbose=False,
                                                      conf=float(os.environ['CONFIDENCE']),
                                                      persist=True,
                                                      tracker="src/tracker.yaml",
                                                      iou=float(os.environ['IOU']),
                                                      imgsz=input_size,
                                                      classes=list_classes,
                                                      agnostic_nms=True,
                                                      vid_stride=False,
                                                      )
        else:
            results_detection = model_detection(frame,
                                                save=False,
                                                stream=True,
                                                verbose=False,
                                                conf=float(os.environ['CONFIDENCE']),
                                                iou=float(os.environ['IOU']),
                                                imgsz=input_size,
                                                classes=list_classes)
        # processing detection results
        list_res_detect = []
        list_time_detect = []
        for r in results_detection:
            list_res_detect.append(r.cpu())
            list_time_detect.append(r.speed)

        time_detect = (list_time_detect[0]['preprocess'] +
                       list_time_detect[0]['inference'] +
                       list_time_detect[0]['postprocess'])
        time_detect /= 1000

        if os.environ['DO_POSE'] == ENV_VAR_TRUE_LABEL:
            infer_time = time_detect + time_pose
        else:
            infer_time = time_detect

        list_objects = generate_objects(DetectedObject, list_res_detect,
                                        labels_dict, os.environ['LIBRARY'])

    return list_objects, list_res_pose, infer_time

def generate_objects(DetectedObject, results_image, labels_dict:dict, library:str)->list:
    """
    takes the raw results of the object detection algorithm and generates a list filled
    with each individual detected object and instantiates a class object with its corresponding
    attributes

    params:
        DetectObject: Class of detected objects, sets bbox coordinates, category, confidence
        results_image: raw results of the object detection algorithm
        labels_dict: dictionary with corresponding labels for categories of objects
        library: library of algorithm to use, either yolox or yolov8
    """
    list_objects = []

    try:
        if library == "supergradients" or library == "yolox":
            for i in range(len(results_image.bbox)):
                if results_image.cls[i] in list_classes:
                    bbox_int = [int(x) for x in results_image.bbox[i]]
                    list_objects.append(DetectedObject(id=labels_dict[results_image.cls[i]][0],
                                                       label_num=results_image.cls[i],
                                                       label_str=labels_dict[results_image.cls[i]][0],
                                                       conf=round(results_image.conf[i],2),
                                                       bbox_xy=bbox_int,
                                                       bbox_wh=get_bbox_xywh(bbox_int)))
        if library == "ultralytics":
            for r in results_image:
                boxes = r.boxes.numpy()
                for box in (x for x in boxes if x is not None):
                    if os.environ['DO_TRACKING'] == ENV_VAR_TRUE_LABEL:
                        id = int(box.id) if box.id is not None else 0
                    else:
                        id = labels_dict[int(box.cls)][0] if box.cls is not None else 0

                    list_objects.append(DetectedObject(id=id,
                                                       label_num=int(box.cls),
                                                       label_str=labels_dict[int(box.cls)][0],
                                                       conf=round(float(box.conf),2),
                                                       bbox_xy=box.xyxy[0].astype(np.int32).tolist(),
                                                       bbox_wh=box.xywh[0].astype(np.int32).tolist()))
    except AttributeError: # in case no object is detected, pass to the next frame
        pass

    return list_objects

def compute_postprocessing(list_objects, results_pose, frame, init_vars, cap)->tuple:
    """
    performs postprocessing phase using list of detected objects obtained with the inference done
    by the object detection model

    params
        list_objects: list of instatiated objects, corresponds to a detected object
        results_pose: results obtained from pose_detection model
        frame: current frame given by opencv
        init_vars: initial values such as empty dicts and lists for time in zone
        cap: capture of source video
    """
    t0 = time.time()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # dictionary to store postprocess results like counting of people
    results_postproc = {}

    # draw bounding boxes
    if os.environ['DO_DRAW_BBOX'] == ENV_VAR_TRUE_LABEL:
        [obj.draw_boxes(frame, init_vars['labels_dict']) for obj in list_objects]
    # draw keypoints
    if os.environ['DO_POSE'] == os.environ['DO_DRAW_KEYPOINTS'] == ENV_VAR_TRUE_LABEL:
        draw_keypoints(frame, results_pose)
    # detect man down
    if os.environ['DO_MAN_DOWN'] == ENV_VAR_TRUE_LABEL:
        [obj.get_is_down(cap, frame, os.environ['SHOW_MAN_DOWN']) for obj in list_objects]

    """
    TODO Tasks that need tracking to be present
    """
    if os.environ['DO_TRACKING'] == ENV_VAR_TRUE_LABEL:
        # number of frames object has spent in scene
        for obj in list_objects:
            init_vars['time_in_scene_dict'][obj.id] += 1
            obj.set_time_in_scene(frame, fps, init_vars)
        # create track history
        [obj.build_track(init_vars) for obj in list_objects]
        # draw tracks
        [obj.draw_tracks(frame) for obj in list_objects]
        # estimate direction
        [obj.estimate_direction(frame, init_vars['labels_dict']) for obj in list_objects]
        # for every person inside an area, count the number of frames spent inside
        if os.environ['DO_TIME_ZONE'] == ENV_VAR_TRUE_LABEL:
            for obj in list_objects:
                obj_is_in_zone = obj.get_is_in_zone(init_vars['zone_poly'])
                if obj_is_in_zone:
                    init_vars['time_in_zone_dict'][obj.id] += 1
                # show time inside zone on top of people's boxes
                obj.set_time_in_zone(frame, fps, init_vars, os.environ['SHOW_TIME_ZONE'])
        # count people entering and leaving a certain area
        if os.environ['DO_ENTER_LEAVE'] == ENV_VAR_TRUE_LABEL:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            door_poly, door2_poly = get_door_polygons(frame, os.environ['DOOR_COORDS'], os.environ['SHOW_ZONE'])
            [obj.enter_leave(frame, width, door_poly, door2_poly, init_vars) for obj in list_objects]
            results_postproc['entering'] = len(init_vars['entering'])
            results_postproc['leaving'] = len(init_vars['leaving'])

    # count objects
    if os.environ['DO_COUNT_OBJECTS'] == ENV_VAR_TRUE_LABEL:
        results_postproc['number_objs'] = count_objs(frame,
                                                     list_objects,
                                                     os.environ['SHOW_COUNT_PEOPLE'])
    # count people in zone
    if os.environ['DO_COUNT_ZONE'] == ENV_VAR_TRUE_LABEL:
        results_postproc['number_people_zone'] = count_zone(cap,
                                                            frame,
                                                            list_objects,
                                                            init_vars,
                                                            os.environ['SHOW_ZONE'])

    # get a list with the individual info of each detected object
    obj_info = []
    for obj in list_objects:
        x = obj.obj_info()
        obj_info.append(x)
    results_postproc['obj_info'] = obj_info

    post_process_time = (time.time()-t0)
    return results_postproc, post_process_time, frame

def send_alert(alerts_dict, list_objects, results_postproc, detection_time_elapsed):
    """
    TODO
    """
    alert_man_down_result = None
    alert_too_many_people_result = None

    # Man down
    if os.environ['DO_MAN_DOWN'] == ENV_VAR_TRUE_LABEL:
        alert_man_down = alerts_dict['man_down']
        number_people_down = sum(1 for obj in list_objects if obj.is_down)
        is_someone_down = number_people_down > 0
        alert_man_down_result, man_down_type = alert_man_down.update(is_someone_down,
                                                                     detection_time_elapsed)
    # Too many people
    if os.environ['DO_COUNT_ZONE'] == ENV_VAR_TRUE_LABEL:
        alert_people = alerts_dict['too_many_people']
        is_too_many_people = results_postproc['number_people_zone'] >= int(os.environ['ALERT_MAX_PEOPLE_ZONE'])
        alert_too_many_people_result, too_many_people_type = alert_people.update(is_too_many_people,
                                                                                 detection_time_elapsed)
    if alert_man_down_result:
        return alert_man_down_result, man_down_type, 0
    elif alert_too_many_people_result:
        return alert_too_many_people_result, too_many_people_type, 1
    else:
        return None, None, -1
