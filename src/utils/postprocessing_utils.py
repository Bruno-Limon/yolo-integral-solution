from datetime import datetime
import cv2
import os
import math
import numpy as np

from config import enable_local_work
enable_local_work()

# global flag to evaluate if environment variables are True
ENV_VAR_TRUE_LABEL = "true"

# list of classes to detect
list_classes = [int(x) for x in os.environ['DETECTED_CLASSES'].split(",")]

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
    count_people, count_bike, count_car = 0, 0, 0
    counter = 0

    if 0 in list_classes:
        count_people = sum(1 for obj in list_objects if obj.label_num == 0)
        counter += 1
    if 1 in list_classes:
        count_bike = sum(1 for obj in list_objects if obj.label_num == 1)
        counter += 1
    if 2 in list_classes:
        count_car = sum(1 for obj in list_objects if obj.label_num == 2)
        counter += 1

    if show_count_people == ENV_VAR_TRUE_LABEL:
        counter_position = 1

        cv2.rectangle(img=frame, pt1=(0,0), pt2=(110,10+(counter*25)), color=(0,0,0), thickness=-1)
        if 0 in list_classes:
            cv2.putText(img=frame, text=f"People: {count_people}", org=(0, counter_position*25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.6, color=(255,255,255), thickness=1)
            counter_position += 1
        if 1 in list_classes:
            cv2.putText(img=frame, text=f"Bycicles: {count_bike}", org=(0, counter_position*25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.6, color=(255,255,255), thickness=1)
            counter_position += 1
        if 2 in list_classes:
            cv2.putText(img=frame, text=f"Cars: {count_car}", org=(0, counter_position*25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.6, color=(255,255,255), thickness=1)
            counter_position += 1

    return [count_people, count_bike, count_car]

def count_zone(cap, frame, list_objects:list, init_vars, show_zone:str)->int:
    """
    performs counting of people inside a given zone
    TODO
    params:
        cap:
        frame: current frame given by opencv
        list_objects: list of instatiated objects, corresponds to a detected object
        init_vars: initial values such as coordinates for the zone or logo for alert
        show_zone: boolean variable to show zone in visualization, from env_vars
    """
    count_people_polygon = 0
    zone_poly = init_vars['zone_poly']
    # cross_poly = init_vars['cross_poly']

    for obj in (obj for obj in list_objects if obj.label_num == 0):
        x, y, w, h = obj.bbox_wh
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(x, y+int((h/2))), measureDist=False)
        if point_in_polygon >= 0:
            count_people_polygon += 1

    if show_zone == ENV_VAR_TRUE_LABEL:
        cv2.polylines(img=frame, pts=[zone_poly], isClosed=True, color=(255,0,0), thickness=2)
        # cv2.polylines(img=frame, pts=[cross_poly], isClosed=True, color=(0,0,255), thickness=2)
        cv2.rectangle(img=frame, pt1=(0,80), pt2=(110,105), color=(0,0,0), thickness=-1)
        cv2.putText(img=frame, text=f"In zone: {count_people_polygon}", org=(0, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.6, color=(255,255,255), thickness=1)

    # overlay graphical alert for too many people in zone
    if count_people_polygon >= int(os.environ['ALERT_MAX_PEOPLE_ZONE']):
        cv2.polylines(img=frame, pts=[zone_poly], isClosed=True, color=(0,0,255), thickness=3)
        overlay_alert(cap, frame, init_vars, "toomanypeople")

    return count_people_polygon


def density_lane(frame, list_objects:list)->int:
    """
    performs counting of people inside a given zone
    TODO
    params:
        cap:
        frame: current frame given by opencv
        list_objects: list of instatiated objects, corresponds to a detected object
        init_vars: initial values such as coordinates for the zone or logo for alert
        show_zone: boolean variable to show zone in visualization, from env_vars
    """
    count_lane_list = [0]*6
    lanes_poly_list = []
    lanes_coord_list = ["524,378|49,717|220,719|568,377",
                        "594,377|563,379|220,719|389,719",
                        "591,377|622,378|558,718|393,718",
                        "871,719|675,379|648,379|698,718",
                        "1046,717|715,376|678,378|874,719",
                        "1046,719|713,379|761,378|1254,718"]
    area_lane_list = []
    lane_density_list = []

    for lane in lanes_coord_list:
        lane_poly = get_zone_poly(lane)
        lanes_poly_list.append(lane_poly)

    for lane in lanes_poly_list:
        # cv2.polylines(img=frame, pts=[lane], isClosed=True, color=(255,0,0), thickness=2)
        area_lane = cv2.contourArea(lane)
        area_lane_list.append(area_lane)

    for obj in list_objects:
        x, y, w, h = obj.bbox_wh
        for i, lane in enumerate(lanes_poly_list):
            point_in_polygon = cv2.pointPolygonTest(contour=lane, pt=(x, y), measureDist=False)
            if point_in_polygon >= 0:
                count_lane_list[i] += 1

    for i in range(6):
        density = count_lane_list[i]/area_lane_list[i]*1000
        lane_density_list.append(density)

    # print(count_lane_list)
    print(lane_density_list)

    return count_lane_list, lane_density_list

def is_congestion(frame, list_objects, count_lane_list, lane_density_list):
    print("x")

def get_door_polygons(frame, door_coords, show_zone):
    """
    performs counting of people inside a given zone
    TODO
    params:
        frame: current frame given by opencv
        door_coords:
        show_zone
    """
    door = [[int(y) for y in x.split(',')] for x in door_coords.split('|')]
    aX, aY = door[0]
    bX, bY = door[1]
    length = cv2.norm(np.array(door[0]), np.array(door[1]))/10

    # get mvt vector
    vX = bX-aX
    vY = bY-aY
    # normalize vector
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    # swapping x and y and inverting one of them
    temp = vX
    swapped_vX = 0-vY
    swapped_vY = temp
    cX = int(bX + swapped_vX * length)
    cY = int(bY + swapped_vY * length)
    dX = int(bX - swapped_vX * length)
    dY = int(bY - swapped_vY * length)
    eX = int(aX + swapped_vX * length)
    eY = int(aY + swapped_vY * length)
    fX = int(aX - swapped_vX * length)
    fY = int(aY - swapped_vY * length)

    # if show_zone == ENV_VAR_TRUE_LABEL:
    #     cv2.line(img=frame, pt1=(aX, aY),  pt2=(bX, bY), color=(0,204,255), thickness=3)
    #     cv2.line(img=frame, pt1=(cX, cY),  pt2=(dX, dY), color=(0,204,255), thickness=3)
    #     cv2.line(img=frame, pt1=(eX, eY),  pt2=(fX, fY), color=(0,204,255), thickness=3)

    door_poly = np.array([(aX, aY),
                          (bX, bY),
                          (cX, cY),
                          (eX, eY)], np.int32).reshape((-1, 1, 2))

    door2_poly = np.array([(fX, fY),
                           (dX, dY),
                           (bX, bY),
                           (aX, aY)], np.int32).reshape((-1, 1, 2))

    return door_poly, door2_poly

def send_frame_info(results_postprocessing:dict, cap)->dict:
    """
    sends scene information including items such as time, camera, frame, number of detected
    objects,etc. As well as individual info of each detected object, such as bbox, conf, category

    params:
        results_postprocessing: dictionary containing postprocessed information such as the time
            spent inside an area
        cap: opencv capture of current frame
    """
    frame_info_dict = {"date_time": datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                       "device_id": "-",
                       "camera_id": "-",
                       "model_id": "-",
                       "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                       "num_people": results_postprocessing['number_objs'][0],
                       "num_bikes": results_postprocessing['number_objs'][1],
                       "num_cars": results_postprocessing['number_objs'][2],
                       "people_in_zone": results_postprocessing['number_people_zone'],
                       "list_objects": results_postprocessing['obj_info']}

    return frame_info_dict

def aggregate_info(list_aggregates:list, results_postprocessing:dict, list_objects:list)->list:
    """
    stores object information in a list that will then be used to aggregate info through time and

    params:
        list_aggregates: empty list to store aggregated info
        results_postprocessing: dictionary containing postprocessed information such as the time
            spent inside an area
        list_objects: list of instatiated objects, corresponds to a detected object
    """
    # number of objects
    if "number_objs" in results_postprocessing:
        for i in range(0,3): list_aggregates[i].append(results_postprocessing['number_objs'][i])
    else:
        for i in range(0,3): list_aggregates[i].append(0)

    # number of people in zone
    if "number_people_zone" in results_postprocessing:
        list_aggregates[3].append(results_postprocessing['number_people_zone'])
    else:
        list_aggregates[3].append(0)

    # count number of people down
    number_people_down = sum(1 for obj in list_objects if obj.is_down)
    if number_people_down:
        list_aggregates[4].append(number_people_down)
    else:
        list_aggregates[4].append(0)

    # time in zone
    time_in_zone = [obj.time_in_zone for obj in list_objects]
    masked_time_in_zone = np.ma.masked_equal(time_in_zone,0).compressed()
    if np.mean(masked_time_in_zone) > 0:
        list_aggregates[5].append(np.max(masked_time_in_zone))
        list_aggregates[5].append(np.min(masked_time_in_zone))
    else:
        list_aggregates[5].append(0)

    # enter / leaving
    if "entering" in results_postprocessing:
        list_aggregates[6].append(results_postprocessing['entering'])
    if "leaving" in results_postprocessing:
        list_aggregates[7].append(results_postprocessing['leaving'])

    # direction
    list_directions = [obj.direction for obj in list_objects]
    unique, counts = np.unique(list_directions, return_counts=True)
    if counts.size:
        index = np.argmax(counts)
        direction_frame = unique[index]
    else:
        direction_frame = "stationary"

    if list_directions:
            list_aggregates[8].append(direction_frame)
    else:
        list_aggregates[8].append("stationary")
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
    #
    if np.mean(list_aggregates[5]) > 0:
         max_time_in_zone = float(np.max(list_aggregates[5]))
         min_time_in_zone = float(np.min(list_aggregates[5]))
         avg_time_in_zone = round(float(np.mean(list_aggregates[5])),1)
    else:
        max_time_in_zone = 0
        min_time_in_zone = 0
        avg_time_in_zone = 0

    avg_people_down = int(np.ceil(np.mean(list_aggregates[4]))) if list_aggregates[4] else 0
    max_people = int(np.max(list_aggregates[0])) if list_aggregates[0] else 0
    min_people = int(np.min(list_aggregates[0])) if list_aggregates[0] else 0
    avg_people = int(np.floor(np.mean(list_aggregates[0]))) if list_aggregates[0] else 0
    avg_bikes = int(np.floor(np.mean(list_aggregates[1]))) if list_aggregates[1] else 0
    avg_cars = int(np.floor(np.mean(list_aggregates[2]))) if list_aggregates[2] else 0
    max_people_in_zone = int(np.max(list_aggregates[3])) if list_aggregates[3] else 0
    min_people_in_zone = int(np.min(list_aggregates[3])) if list_aggregates[3] else 0
    avg_people_in_zone = int(np.floor(np.mean(list_aggregates[3]))) if list_aggregates[3] else 0
    sum_entrances = int(np.max(list_aggregates[6])) if list_aggregates[6] else 0
    sum_exits = int(np.max(list_aggregates[7])) if list_aggregates[7] else 0

    unique, counts = np.unique(list_aggregates[8], return_counts=True)
    if counts.size:
        idx = np.argmax(counts)
        direction = unique[idx]
    else:
        direction = "stationary"

    agg_frame_info_dict = {"message_type": "MESSAGE",
                           "interval_id": time_interval_counter,
                           "interval_start": time_interval_start,
                           "interval_end": time_interval_end,
                           "avg_people_down": avg_people_down,
                           "max_people": max_people,
                           "min_people": min_people,
                           "avg_people": avg_people,
                           "avg_bikes": avg_bikes,
                           "avg_cars": avg_cars,
                           "max_people_in_zone": max_people_in_zone,
                           "min_people_in_zone": min_people_in_zone,
                           "avg_people_in_zone": avg_people_in_zone,
                           "max_time_in_zone": max_time_in_zone,
                           "min_time_in_zone": min_time_in_zone,
                           "avg_time_in_zone": avg_time_in_zone,
                           "sum_entrances": sum_entrances,
                           "sum_exits": sum_exits,
                           "main_direction": direction,
                           "device_id": "-",
                           "camera_id": "-",
                           "model_id": "-"}

    return agg_frame_info_dict

def send_alert_info(alert_type, alert_id):
    alert_info_dict = {"message_type": "ALERT",
                       "alert_id": alert_id,
                       "device_id": "-",
                       "time": datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                       "alert_type": alert_type}

    return alert_info_dict

def print_fps(frame, frame_width:int, frame_height:int, times_dict:dict)->None:
    """
    computes frames per second and prints it

    params:
        frame: current frame
        frame_width: width
        frame_height: height
        times_dict: dict with processing times
    """
    print(f"Read frame time: {round(times_dict['read_frame_time'], 4)}")
    print(f"inference time: {round(times_dict['infer_time'], 4)}")
    print(f"Postprocessing time: {round(times_dict['post_process_time'], 4)}")
    print(f"total process time: {round(times_dict['total_time'], 4)}")

    current_fps = round((1 / times_dict['total_time']), 2)
    print(f"fps: {current_fps} \n")

    cv2.rectangle(img=frame, pt1=(frame_width - 60, frame_height - 35),
                  pt2=(frame_width, frame_height), color=(0,0,0), thickness=-1)
    cv2.putText(img=frame, text=str(current_fps), org=(frame_width - 55, frame_height - 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255,255,255), thickness=1)

def draw_keypoints(frame, results_pose):
    """
    ---

    BODY_MAP = {"nose": 0, "eye_sx": 1, "eye_dx": 2, "ear_sx": 3,
                "ear_dx": 4, "shoulder_sx": 5, "shoulder_dx": 6,
                "elbow_sx": 7, "elbow_dx": 8, "wrist_sx": 9,
                "wrist_dx": 10, "hip_sx": 11, "hip_dx": 12,
                "knee_sx": 13, "knee_dx": 14, "foot_sx": 15, "foot_dx": 16}
    params:
        frame:
        results_pose:
    """
    color_saggital = (100,0,255)
    color_head = (175, 255, 46, .5)
    color_torso = (54, 104, 206, .5)
    color_arm_sx = (255, 0, 255, .5)
    color_arm_dx = (0, 200, 255, .5)
    color_leg_sx = (0, 0, 200, .5)
    color_leg_dx = (252, 168, 0, .5)
    tkn = 2 # thickness

    def find_middle_point(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2
        return (int(middle_x), int(middle_y))

    def circle_kp(color):
        cv2.circle(img=frame, center=(kp[i][j][0], kp[i][j][1]),
                   radius=2, color=color, thickness=tkn)
    def line_hip_shoulder(color):
        cv2.line(img=frame, pt1=(kp[i][j][0], kp[i][j][1]),
                 pt2=(kp[i][j+1][0], kp[i][j+1][1]), color=color, thickness=tkn)
    def line_ear_shoulder(color):
        cv2.line(img=frame, pt1=(kp[i][j][0], kp[i][j][1]),
                 pt2=(kp[i][j+2][0], kp[i][j+2][1]), color=color, thickness=tkn)
    def connect_line(color):
        cv2.line(img=frame, pt1=(kp[i][j][0], kp[i][j][1]),
                 pt2=(kp[i][j-2][0], kp[i][j-2][1]), color=color, thickness=tkn)

    for r in results_pose: kp = r.keypoints.xy.numpy().astype(np.int32)
    if kp.size != 0:
        for i in range(len(kp)):
            nose = (kp[i][0][0], kp[i][0][1])
            mid_feet = find_middle_point(kp[i][15], kp[i][16])
            cv2.circle(img=frame, center=nose, radius=2, color=color_saggital, thickness=2)
            cv2.line(img=frame, pt1=nose, pt2=mid_feet, color=color_saggital, thickness=3)
            cv2.circle(img=frame, center=mid_feet, radius=2, color=color_saggital, thickness=2)

            for j in range(17):
                # head
                if j in [3,4]: circle_kp(color_head), line_ear_shoulder(color_head)
                # left arm
                if j == 5: circle_kp(color_arm_dx), line_hip_shoulder(color_torso)
                if j in [7,9]: circle_kp(color_arm_dx), connect_line(color_arm_dx)
                # right arm
                if j == 6: circle_kp(color_arm_dx)
                if j in [8,10]: circle_kp(color_arm_dx), connect_line(color_arm_dx)
                # left leg
                if j == 11: circle_kp(color_leg_dx), line_hip_shoulder(color_torso)
                if j in [13,15]: circle_kp(color_leg_dx), connect_line(color_leg_dx)
                # right leg
                if j == 12: circle_kp(color_leg_dx)
                if j in [14,16]: circle_kp(color_leg_dx), connect_line(color_leg_dx)
                # sagittal plane



            cv2.line(img=frame, pt1=(kp[i][6][0], kp[i][6][1]),
                     pt2=(kp[i][12][0], kp[i][12][1]), color=color_torso, thickness=tkn)
            cv2.line(img=frame, pt1=(kp[i][5][0], kp[i][5][1]),
                     pt2=(kp[i][11][0], kp[i][11][1]), color=color_torso, thickness=tkn)

"""TODO WORK IN PROGRESS TODO"""
def draw_nearest_points(frame, bboxes):
    def get_nearest_points(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        points_bbox1 = np.array([(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)])
        points_bbox2 = np.array([(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)])

        distances = np.linalg.norm(points_bbox1[:, np.newaxis, :] - points_bbox2, axis=2)
        min_indices = np.unravel_index(np.argmin(distances), distances.shape)

        nearest_point_bbox1 = tuple(points_bbox1[min_indices[0]])
        nearest_point_bbox2 = tuple(points_bbox2[min_indices[1]])

        return nearest_point_bbox1, nearest_point_bbox2

    # Draw lines connecting nearest points
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            x1, y1, w1, h1 = bboxes[i]
            x2, y2, w2, h2 = bboxes[j]
            nearest_point1, nearest_point2 = get_nearest_points(bboxes[i], bboxes[j])
            distance = np.sqrt((nearest_point2[0] - nearest_point1[0])**2 +
                               (nearest_point2[1] - nearest_point1[1])**2)
            # print(i,j,distance)
            # cv2.line(frame, nearest_point1, nearest_point2, (0, 255, 0), 2)
            # if distance < 80:
            #     # cv2.line(frame, nearest_point1, nearest_point2, (0, 255, 255), 2)
            if distance < 50:
                cv2.line(frame, nearest_point1, nearest_point2, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 3)
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 3)

                # # header of bbox to put info into
                # cv2.rectangle(img=frame,
                #             pt1=(self.bbox_x1 - 1, self.bbox_y1 - int(1.5 * txt_size[1])),
                #             pt2=(self.bbox_x1 + txt_size[0] + 2, self.bbox_y1),
                #             color=labels_dict[int(self.label_num)][1],
                #             thickness=-1)

                # # information about object, id and conf
                # cv2.putText(img=frame,
                #             text=text,
                #             org=(self.bbox_x1, self.bbox_y1 - int(.5 * txt_size[1])),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=.5,
                #             color=(255,255,255),
                #             thickness=1)

def overlay_alert(cap, frame, init_vars:dict, alert_type:str):
    """
    TODO
    """
    # where to insert logo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logo_x, logo_y = init_vars['zone_poly'][0][0]
    logo_x = (width-logo_x)
    logo_y = (height-logo_y)
    logo_size = init_vars['logo_size']

    if alert_type == "toomanypeople":
        roi = frame[-logo_size-logo_y+int(logo_size/2):-logo_y+int(logo_size/2),
                    -logo_size-logo_x+int(logo_size/2):-logo_x+int(logo_size/2)]
    elif alert_type == "mandown":
        roi = frame[-logo_size-logo_y+int(logo_size/2):-logo_y+int(logo_size/2),
                    -logo_size-logo_x+int(logo_size/2):-logo_x+int(logo_size/2)]
    else:
        list_alert_type = ["toomanypeople", "mandown"]
        raise ValueError(f"Error: alert_type = {alert_type} must be in {list_alert_type}")

    # set an index of where the mask is
    roi[np.where(init_vars[f'mask_{alert_type}'])] = 0
    roi += init_vars[f'logo_{alert_type}']

def get_labels_dict()->dict:
    """
    returns a dictionary containing the corresponding labels for the possible categories to detect,
    their number, name and BGR color for visualization
    """
    labels_dict = {0: ["person", (209,209,0)],
                   1: ["bicycle", (47,139,237)],
                   2: ["car", (42,237,139)],
                #    2: ["car", (200,50,0)],
                   3: ["motorcycle", (56,0,255)],
                   5: ["bus", (169,10,150)],
                   7: ["truck", (169,255,143)]}

    return labels_dict
