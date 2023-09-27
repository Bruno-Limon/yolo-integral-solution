from datetime import datetime
import cv2
import numpy as np
import json


def get_labels_dict():
    labels_dict = {0: ["person", (209,209,0)],
                   1: ["bicycle", (47,139,237)],
                   2: ["car", (42,237,139)]}

    return labels_dict

def get_zone_poly(zone_coords):

    # turning string into list with coordinates
    polygon = [[int(y) for y in x.split(',')] for x in zone_coords.split('|')]

    # zone to count people in
    zone_poly = np.array([polygon[0], # left upper corner
                          polygon[1], # right upper corner
                          polygon[2], # right lower corner
                          polygon[3]], np.int32) # left lower corner
    zone_poly = zone_poly.reshape((-1, 1, 2))

    return zone_poly

def get_door_thresholds(door_coords, door2_coords):
    # first threshold of "door"
    polygon = [[int(y) for y in x.split(',')] for x in door_coords.split('|')]
    polygon2 = [[int(y) for y in x.split(',')] for x in door2_coords.split('|')]

    door_poly = np.array([polygon[0],
                          polygon[1],
                          polygon[2],
                          polygon[3]], np.int32)
    door_poly = door_poly.reshape((-1, 1, 2))

    # second threshold
    door2_poly = np.array([polygon2[0],
                           polygon2[1],
                           polygon2[2],
                           polygon2[3]], np.int32)
    door2_poly = door2_poly.reshape((-1, 1, 2))

    return door_poly, door2_poly

def get_bbox_xywh(bbox_xy):
    x1, y1, x2, y2 = bbox_xy
    center = [int((x1+x2)/2), int((y1+y2)/2)]
    w = x2-x1
    h = y2-y1

    return [center[0], center[1], w, h]

# counting overall objects on screen, including people, bikes and cars
def count_objs(frame, list_objects, show_count_people):
    count_people = sum(1 for obj in list_objects if obj.label_num == 0)
    count_bike = sum(1 for obj in list_objects if obj.label_num == 1)
    count_car = sum(1 for obj in list_objects if obj.label_num == 2)

    if show_count_people == True:
        cv2.rectangle(img=frame, pt1=(0,0), pt2=(110,80), color=(0,0,0), thickness=-1)
        cv2.putText(img=frame, text=f"People: {count_people}", org=(0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Bicycles: {count_bike}", org=(0, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Cars: {count_car}", org=(0, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.6, color=(255,255,255), thickness=1)

    return [count_people, count_bike, count_car]

def count_zone(frame, list_objects, zone_poly, show_zone):
    count_people_polygon = 0

    if show_zone == True:
        cv2.polylines(img=frame, pts=[zone_poly], isClosed=True, color=(255,0,0), thickness=2)

    for obj in (obj for obj in list_objects if obj.label_num == 0):
        x, y, w, h = obj.bbox_wh
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(x, y+int((h/2))), measureDist=False)
        if point_in_polygon >= 0:
            count_people_polygon += 1

    return count_people_polygon

# dynamically creating objects from yolo detection and pose estimation
def generate_objects(DetectedObject, results_image, labels_dict):
    list_objects = []

    # object detection results
    try:
        for i in range(len(results_image.bbox)):
            if results_image.cls[i] in [0, 1, 2]:
                list_objects.append(DetectedObject(id=i+1,
                                                   label_num=results_image.cls[i],
                                                   label_str=labels_dict[results_image.cls[i]][0],
                                                   conf=round(results_image.conf[i],2),
                                                   bbox_xy=results_image.bbox[i],
                                                   bbox_wh=get_bbox_xywh(results_image.bbox[i])
                                                  )
                                    )
    except AttributeError:
        pass

    return list_objects

def send_frame_info(number_objs, number_people_zone, cap, obj_info):
    frame_info_dict = {"date_time": datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                       "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                       "num_people": number_objs[0],
                       "num_bikes": number_objs[1],
                       "num_cars": number_objs[2],
                       "people_in_zone": number_people_zone,
                       "people_enter": 0,
                       "people_leave": 0,
                       "list_objects": obj_info}

    json_obj = json.dumps(obj=frame_info_dict, indent=4)
    return json_obj

def print_fps(frame, frame_width, frame_height, infer_time, process_time):
    print(f"inference time: {round(infer_time, 4)}")
    print(f"process time: {round(process_time, 4)}")
    current_fps = round((1 / process_time), 2)
    print(f"fps: {current_fps}")

    cv2.rectangle(img=frame, pt1=(frame_width - 50, frame_height - 35),
                  pt2=(frame_width, frame_height), color=(0,0,0), thickness=-1)
    cv2.putText(img=frame, text=str(current_fps), org=(frame_width - 45, frame_height - 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255,255,255), thickness=1)
