from datetime import datetime
import cv2
import numpy as np

# global flag to evaluate if environment variables are True
ENV_VAR_TRUE_LABEL = "true"

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
                if results_image.cls[i] in [0,1,2,3,5,7]:
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

                # only creating an object if it belongs to the chosen classes
                for box in (x for x in boxes if x.cls in [0,1,2,3,5,7]):
                    if box.id != None:
                        list_objects.append(DetectedObject(id=int(box.id),
                                                           label_num=int(box.cls),
                                                           label_str=labels_dict[int(box.cls)][0],
                                                           conf=round(float(box.conf),2),
                                                           bbox_xy=box.xyxy[0].astype(np.int32).tolist(),
                                                           bbox_wh=box.xywh[0].astype(np.int32).tolist()))
    except AttributeError: # in case no object is detected, pass to the next frame
        pass

    return list_objects

def get_bbox_xywh(bbox_xy:list)->list:
    """
    Takes the raw output of bounding boxes in xyxy format and calculates the center,
    width and height of the bounding box.

    params
        bbox_xy: list of coordinates for bounding box in format [x1,y1,x2,y2]
    """
    x1, y1, x2, y2 = bbox_xy
    center = [int((x1+x2)/2), int((y1+y2)/2)]
    w = x2-x1
    h = y2-y1

    return [center[0], center[1], w, h]

def get_zone_poly(zone_coords:str)->np.array:
    """
    takes environment variable zone_coords and turns the str into a suitable polygon
    to be used as an area of interest.

    params
        zone_coords: string with coordinates in format x1,y1|x2,y2|x3,y3|x4,y4
    """
    # turning string into list with coordinates
    polygon = [[int(y) for y in x.split(',')] for x in zone_coords.split('|')]

    # zone to count people in
    zone_poly = np.array([polygon[0], # left upper corner
                          polygon[1], # right upper corner
                          polygon[2], # right lower corner
                          polygon[3]], np.int32) # left lower corner
    zone_poly = zone_poly.reshape((-1, 1, 2))

    return zone_poly

# counting overall objects on screen, including people, bikes and cars
def count_objs(frame, list_objects:list, show_count_people:str)->list:
    """
    performs counting of the different classes present in the video, such as people, bikes, etc.

    params:
        frame: current frame given by opencv
        list_objects: list of instatiated objects, corresponds to a detected object
        show_count_people: boolean variable to show count on screen, from env_vars
    """
    count_people = sum(1 for obj in list_objects if obj.label_num == 0)
    count_bike = sum(1 for obj in list_objects if obj.label_num == 1)
    count_car = sum(1 for obj in list_objects if obj.label_num == 2)

    if show_count_people == ENV_VAR_TRUE_LABEL:
        cv2.rectangle(img=frame, pt1=(0,0), pt2=(110,80), color=(0,0,0), thickness=-1)
        cv2.putText(img=frame, text=f"People: {count_people}", org=(0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Bicycles: {count_bike}", org=(0, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Cars: {count_car}", org=(0, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.6, color=(255,255,255), thickness=1)

    return [count_people, count_bike, count_car]

def count_zone(frame, list_objects:list, zone_poly, door_coords, show_zone:str)->int:
    """
    performs counting of people inside a given zone

    params:
        frame: current frame given by opencv
        list_objects: list of instatiated objects, corresponds to a detected object
        zone_poly: np.array with coordinates for the zone of interest
        show_zone: boolean variable to show zone in visualization, from env_vars
    """
    count_people_polygon = 0
    #door = [[int(y) for y in x.split(',')] for x in door_coords.split('|')]

    if show_zone == ENV_VAR_TRUE_LABEL:
        cv2.polylines(img=frame, pts=[zone_poly], isClosed=True, color=(255,0,0), thickness=2)
        #cv2.line(img=frame, pt1=(door[0]),  pt2=(door[1]), color=(0,204,255), thickness=3)

    for obj in (obj for obj in list_objects if obj.label_num == 0):
        x, y, w, h = obj.bbox_wh
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(x, y+int((h/2))), measureDist=False)
        if point_in_polygon >= 0:
            count_people_polygon += 1

    return count_people_polygon

def send_frame_info(number_objs:list, number_people_zone:int, cap, obj_info:dict)->dict:
    """
    sends scene information including items such as time, camera, frame, number of detected
    objects,etc. As well as individual info of each detected object, such as bbox, conf, category

    params:
        number_objs: list containing counts of detected objects
        number_people_zone: number of people in zone
        cap: opencv capture of current frame
        obj_info: list containing
    """
    frame_info_dict = {"date_time": datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                       "device_id": "-",
                       "camera_id": "-",
                       "model_id": "-",
                       "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                       "num_people": number_objs[0],
                       "num_bikes": number_objs[1],
                       "num_cars": number_objs[2],
                       "people_in_zone": number_people_zone,
                       "people_enter": 0,
                       "people_leave": 0,
                       "list_objects": obj_info}

    return frame_info_dict

def aggregate_info(list_aggregates:list, number_objs:list, number_people_zone:int, list_objects:list)->list:
    """
    stores object information in a list that will then be used to aggregate info through time and

    params:
        list_aggregates: empty list to store aggregated info
        number_objs: list containing counts of detected objects
        number_people_zone: number of people in zone
        list_objects: list of instatiated objects, corresponds to a detected object
    """
    # number of objects
    for i in range(0,3):
        list_aggregates[i].append(number_objs[i])

    # number of people in zone
    list_aggregates[3].append(number_people_zone)

    # count number of people down
    number_people_down = sum(1 for obj in list_objects if obj.is_down)
    list_aggregates[4].append(number_people_down)
    # print(list_aggregates[4])

    return list_aggregates

def send_agg_frame_info(list_aggregates:list, time_interval_start:datetime,
                        time_interval_end:datetime, time_interval_counter:int)->dict:
    """
    instead of sending info about every frame, it takes the stored info across an interval
    of time and aggregates it to send every x time inverval

    params:
        list_aggregates: empty list to store aggregated info
        time_interval_start: time at the beginning of the interval
        time_interval_end: time at the end of the interval
        time_interval_counter: the current interval
    """
    agg_frame_info_dict = {"interval_id": time_interval_counter,
                           "interval_start": time_interval_start,
                           "interval_end": time_interval_end,
                           "avg_people_down": np.ceil(np.mean(list_aggregates[4])),
                           "avg_people": np.floor(np.mean(list_aggregates[0])),
                           "avg_bikes": np.floor(np.mean(list_aggregates[1])),
                           "avg_cars": np.floor(np.mean(list_aggregates[2])),
                           "avg_people_in_zone": np.floor(np.mean(list_aggregates[3])),
                           "avg_time_in_zone": 0,
                           "sum_entrances": 0,
                           "sum_exits": 0,
                           "device_id": "-",
                           "camera_id": "-",
                           "model_id": "-"}

    return agg_frame_info_dict

def print_fps(frame, frame_width:int, frame_height:int, infer_time:float, process_time:float)->None:
    """
    computes frames per second and prints it

    params:
        frame: current frame
        frame_width: width
        frame_height: height
        infer_time: time for the detection algorithm to return raw results
        process_time: time to process an entire frame
    """
    print(f"inference time: {round(infer_time, 4)}")
    print(f"process time: {round(process_time, 4)}")

    current_fps = round((1 / process_time), 2)
    print(f"fps: {current_fps} \n")

    cv2.rectangle(img=frame, pt1=(frame_width - 50, frame_height - 35),
                  pt2=(frame_width, frame_height), color=(0,0,0), thickness=-1)
    cv2.putText(img=frame, text=str(current_fps), org=(frame_width - 45, frame_height - 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255,255,255), thickness=1)

def load_model_size(model_size:str, library:str)->str:
    """
    given the size and library of models to use, returns the correct way to
    initialize their model

    params:
        model_size: size of model to use, either nano, tiny, small or medium
        library: library to use either yolox or yolov8
    """
    if library == "supergradients":
        if model_size == "nano": model = "yolox_n"
        if model_size == "tiny": model = "yolox_t"
        if model_size == "small": model = "yolox_s"
        if model_size == "medium": model = "yolox_m"

    if library == "yolox":
        if model_size == "nano": model = "src/models/yolox_nano.pth"
        if model_size == "tiny": model = "src/models/yolox_tiny.pth"
        if model_size == "small": model = "src/models/yolox_s.pth"
        if model_size == "medium": model = "src/models/yolox_m.pth"

    if library == "ultralytics":
        if model_size == "nano": model = "yolo_nas_s.pt"
        # if model_size == "nano": model = "yolov8n.pt"
        if model_size == "small": model = "yolov8s.pt"
        if model_size == "medium": model = "yolov8m.pt"

    return model

def get_labels_dict()->dict:
    """
    returns a dictionary containing the corresponding labels for the possible categories to detect,
    their number, name and BGR color for visualization
    """
    labels_dict = {0: ["person", (209,209,0)],
                   1: ["bicycle", (47,139,237)],
                   2: ["car", (42,237,139)],
                   3: ["motorcycle", (56,0,255)],
                   5: ["bus", (169,10,150)],
                   7: ["truck", (169,255,143)]}

    return labels_dict

# def get_door_thresholds(door_coords, door2_coords):
#     # first threshold of "door"
#     polygon = [[int(y) for y in x.split(',')] for x in door_coords.split('|')]
#     polygon2 = [[int(y) for y in x.split(',')] for x in door2_coords.split('|')]

#     door_poly = np.array([polygon[0],
#                           polygon[1],
#                           polygon[2],
#                           polygon[3]], np.int32)
#     door_poly = door_poly.reshape((-1, 1, 2))

#     # second threshold
#     door2_poly = np.array([polygon2[0],
#                            polygon2[1],
#                            polygon2[2],
#                            polygon2[3]], np.int32)
#     door2_poly = door2_poly.reshape((-1, 1, 2))

#     return door_poly, door2_poly