from datetime import datetime
from collections import defaultdict
import os
from dotenv import load_dotenv

from ultralytics import YOLO
import numpy as np
import torch
import cv2
import gc
import json


# enable rtsp capture for opencv
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# text settings
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2

# dictionary to map the class number obtained with yolo with its name and color for bounding boxes
labels_dict = {0: ["person", (209,209,0)],
               1: ["bicycle", (47,139,237)],
               2: ["car", (42,237,139)]}

# zone to count people in
zone_poly = np.array([[480, 160], #x1, y1 = left upper corner
                      [780, 200], #x2, y2 = right upper corner
                      [580, 380], #x3, y3 = right lower corner
                      [200, 280]], np.int32) #x4, y4 = left lower corner
zone_poly = zone_poly.reshape((-1, 1, 2))

# dictionaries containing id of people entering/leaving and sets to count them
people_entering_dict = {}
entering = set()
people_leaving_dict = {}
leaving = set()

#first threshold of "door"
door_poly = np.array([[200, 280], #x1, y1 = left upper corner
                      [440, 355], #x2, y2 = right upper corner
                      [430, 365], #x3, y3 = right lower corner
                      [185, 290]], np.int32) #x4, y4 = left lower corner
door_poly = door_poly.reshape((-1, 1, 2))

# second threshold
door2_poly = np.array([[180, 300], #x1, y1 = left upper corner
                       [420, 375], #x2, y2 = right upper corner
                       [410, 385], #x3, y3 = right lower corner
                       [165, 310]], np.int32) #x4, y4 = left lower corner
door2_poly = door2_poly.reshape((-1, 1, 2))


# class to instantiate every object detected and its attributes,
# in case of car or bicycle set "keypoints" to 0
class DetectedObject:

    def __init__(self, id, label_num, label_str, conf, bbox_xy, bbox_wh):
        self.id = id
        self.label_num = label_num
        self.label_str = label_str
        self.conf = conf
        self.bbox_xy = bbox_xy # left upper corner and right lower corner
        self.bbox_wh = bbox_wh # center of bbox together with width and height
        self.is_down = False
        self.is_in_zone = False
        self.time_in_zone = 0
        self.info = {}

    # draw each bounding box with the color selected in "labels_dict", writes id and confidence of detected object
    def draw_boxes(self, frame):

        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        cv2.rectangle(img=frame, pt1=(x1,y1), pt2=(x2,y2),
                      color=labels_dict[int(self.label_num)][1], thickness=thickness)
        cv2.rectangle(img=frame, pt1=(x1-thickness, y1-20), pt2=(x2+thickness, y2-h),
                      color=labels_dict[int(self.label_num)][1], thickness=-1)

        # central lower point of bounding box
        cv2.circle(img=frame, center=(x, y+int((h/2))), radius=4, color=(0,255,0), thickness=-1)

        # information about each object, can contain any class attribute
        cv2.putText(img=frame, text=f"id:{self.id} {self.conf}", org=(x1, y1-5), fontFace=font,
                    fontScale=.5, color=(255,255,255), thickness=1)

    # draw a line following the movement of detected people during a certain amount of frames
    def draw_tracks(self, frame, track_history_dict):

        if self.label_num == 0:
            x, y, w, h = self.bbox_wh
            track = track_history_dict[self.id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 60:  # how many frames to keep the track
                track.pop(0)

            # draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(img=frame, pts=[points], isClosed=False, color=(230,53,230), thickness=5)

    # detect man down, using the width and height of bounding box, if w > h, then man_down
    def get_is_down(self, frame):

        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        # the angle determines if the person is on the ground or not
        if w > h and self.label_num == 0:
            self.is_down = True
            cv2.rectangle(img=frame, pt1=(x1-thickness, y1+(h-50)), pt2=(x2+thickness, y2),
                        color=(0,0,0), thickness=-1)
            cv2.putText(img=frame, text="FALLEN", org=(x1, y2-5), fontFace=font,
                        fontScale=1, color=(255,255,255), thickness=1)

    # determines if person is inside a defined area or not
    def get_is_in_zone(self, poly):

        x, y, w, h = self.bbox_wh

        point_in_polygon = cv2.pointPolygonTest(contour=poly, pt=(x, y+int((h/2))), measureDist=False)
        if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            self.is_in_zone = True

        return self.id, self.is_in_zone

    # shows amount of time a person has spent inside area
    def draw_time_zone(self, frame, poly, time_in_zone_dict, fps):

        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        point_in_polygon = cv2.pointPolygonTest(contour=poly, pt=(x, y+int((h/2))), measureDist=False)
        if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            time = time_in_zone_dict[self.id] # access dict with id and amount of time in zone
            time = round(time/fps, 1)
            cv2.rectangle(img=frame, pt1=(x1-thickness, y1-40), pt2=(x2+thickness, y2-(h+20)),
                            color=(0,0,0), thickness=-1)
            cv2.putText(img=frame, text=str(time), org=(x1+int((w/3)), y1-25), fontFace=font,
                                fontScale=.5, color=(255,255,255), thickness=2)

            self.time_in_zone = time

    # count number of people entering and leaving
    def enter_leave(self, frame, frame_width):

        x, y, w, h = self.bbox_wh

        # lower point of bounding box, representing the feet
        feet_bb = (x, y+int((h/2)))

        # first threshold of door
        cv2.polylines(img=frame, pts=[door_poly], isClosed=True, color=(0,204,255), thickness=thickness)
        # second threshold
        cv2.polylines(img=frame, pts=[door2_poly], isClosed=True, color=(0,204,255), thickness=thickness)

        # people entering
        # if feet of person enter 1st threshold, add them to dict of people entering
        point_in_polygon = cv2.pointPolygonTest(contour=door2_poly, pt=feet_bb, measureDist=False)
        if point_in_polygon >=0:
            people_entering_dict[self.id] = feet_bb

        # when they step on 2nd threshold, add them to the set
        if self.id in people_entering_dict:
            point_in_polygon2 = cv2.pointPolygonTest(contour=door_poly, pt=feet_bb, measureDist=False)
            if point_in_polygon2 >=0:
                entering.add(self.id)

        # people leaving
        # same logic as before but reversing the order of the thresholds
        point_in_polygon3 = cv2.pointPolygonTest(contour=door_poly, pt=feet_bb, measureDist=False)
        if point_in_polygon3 >=0:
            people_leaving_dict[self.id] = feet_bb
        if self.id in people_leaving_dict:
            point_in_polygon4 = cv2.pointPolygonTest(contour=door2_poly, pt=feet_bb, measureDist=False)
            if point_in_polygon4 >=0:
                leaving.add(self.id)

        cv2.rectangle(img=frame, pt1=(frame_width-110,0), pt2=(frame_width,55), color=(0,0,0), thickness=-1)
        cv2.putText(img=frame, text=f"Entering: {len(entering)}", org=(frame_width-110, 25), fontFace=font,
                fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Leaving: {len(leaving)}", org=(frame_width-110, 50), fontFace=font,
                fontScale=.6, color=(255,255,255), thickness=1)

    # prints or returns all info about the detected objects in each frame
    def obj_info(self, number_objs, number_people_zone):

        self.info = {"date_time": datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                     "id": self.id,
                     "class": self.label_str,
                     "confidence": self.conf,
                     "bbox_xywh": self.bbox_wh,
                     "is_down": self.is_down,
                     "is_in_zone": self.is_in_zone,
                     "time_in_zone": self.time_in_zone,
                     "num_people": number_objs[0],
                     "num_bikes": number_objs[1],
                     "num_cars": number_objs[2],
                     "people_in_zone": number_people_zone,
                     "people_enter": len(entering),
                     "people_leave": len(leaving)
        }

        json_obj = json.dumps(obj=self.info, indent=4)
        return json_obj

# counting overall objects on screen, including people, bikes and cars
def count_objs(frame, list_objects):

    count_people = sum(1 for obj in list_objects if obj.label_num == 0)
    count_bike = sum(1 for obj in list_objects if obj.label_num == 1)
    count_car = sum(1 for obj in list_objects if obj.label_num == 2)

    cv2.rectangle(img=frame, pt1=(0,0), pt2=(110,80), color=(0,0,0), thickness=-1)
    cv2.putText(img=frame, text=f"People: {count_people}", org=(0, 25), fontFace=font,
                fontScale=.6, color=(255,255,255), thickness=1)
    cv2.putText(img=frame, text=f"Bicycles: {count_bike}", org=(0, 50), fontFace=font,
                fontScale=.6, color=(255,255,255), thickness=1)
    cv2.putText(img=frame, text=f"Cars: {count_car}", org=(0, 75), fontFace=font,
                fontScale=.6, color=(255,255,255), thickness=1)

    return [count_people, count_bike, count_car]

def count_zone(frame, list_objects):

    count_people_polygon = 0

    for obj in (obj for obj in list_objects if obj.label_num == 0):
        x, y, w, h = obj.bbox_wh
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(x, y+int((h/2))), measureDist=False)
        if point_in_polygon >= 0:
            count_people_polygon += 1

        cv2.polylines(img=frame, pts=[zone_poly], isClosed=True, color=(255,0,0), thickness=thickness)

    return count_people_polygon

# dynamically creating objects from yolo detection and pose estimation
def generate_objects(results_obj):

    list_objects = []

    # object detection results
    for r in results_obj:
        boxes = r.boxes.numpy()

        # only creating an object if it belongs to the chosen classes
        for box in (x for x in boxes if x.cls in [0,1,2]):
            if box.id != None:
                list_objects.append(DetectedObject(id=int(box.id),
                                                   label_num=int(box.cls),
                                                   label_str=labels_dict[int(box.cls)][0],
                                                   conf=round(float(box.conf),2),
                                                   bbox_xy=box.xyxy[0].astype(np.int32).tolist(),
                                                   bbox_wh=box.xywh[0].astype(np.int32).tolist()
                                                   )
                                    )
    return list_objects


# feed the video soruce and apply yolo models, then call the chosen functions for the different tasks as needed
def detect(vid_path, show_image, save_video):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model_obj = YOLO("yolov8n.pt") # tracking and object detection

    LOG_KW = "LOG"
    print(f"{LOG_KW}: model loaded, starting detection")

    # store the track history
    track_history_dict = defaultdict(lambda: [])

    # store the amount of frames spent inside zone
    time_in_zone_dict = defaultdict(int)

    frame_counter = 0
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_video == True:
        output = cv2.VideoWriter("output-demo.avi", cv2.VideoWriter_fourcc(*'MPEG'),
                                 fps=fps, frameSize=(width, height))

    while cap.isOpened():
        success, frame = cap.read()
        frame_counter += fps/2 # every value corresponding to the vid's fps advances the frame by 1 sec
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        if success:
            print(f"{LOG_KW}: video detected")
            results_obj = model_obj.track(frame, save=False, stream=True, verbose=False, conf=.4,
                                          persist=True, tracker="botsort.yaml", iou=.5, classes=[0,1,2])
            results_obj = [x.cpu() for x in results_obj]
            list_objects = generate_objects(results_obj)

            # show bounding boxes
            [obj.draw_boxes(frame) for obj in list_objects]

            # draw tracks
            [obj.draw_tracks(frame, track_history_dict) for obj in list_objects]

            # detect man down
            [obj.get_is_down(frame) for obj in list_objects]

            # for every person inside an area, count the number of frames
            for obj in list_objects:
                obj_id, obj_is_in_zone, = obj.get_is_in_zone(zone_poly)
                if obj_is_in_zone:
                    time_in_zone_dict[obj_id] += 1

            # count objects
            number_objs = count_objs(frame, list_objects)

            # count people in zone
            number_people_zone = count_zone(frame, list_objects)

            # count people entering and leaving a certain area
            [obj.enter_leave(frame, width) for obj in list_objects]

            # show time inside zone on top of people's boxes
            [obj.draw_time_zone(frame, zone_poly, time_in_zone_dict, fps) for obj in list_objects]

            # get object info
            obj_info = []
            for obj in list_objects:
                x = obj.obj_info(number_objs, number_people_zone)
                obj_info.append(x)
            print(f"{LOG_KW}: results:", "\n")
            yield obj_info

            # write output video
            if save_video == True:
                output.write(frame)

            # display the annotated frame
            if show_image == True:
                cv2.imshow("Demo", frame)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # break the loop if the end of the video is reached (comment out when using stream)
        else:
            break

    # release the video capture object and close the display window
    cap.release()

    if save_video == True:
        output.release()

    cv2.destroyAllWindows()

    del model_obj
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # load environment variables
    load_dotenv('../.env')
    VIDEO_SOURCE = os.getenv(key='VIDEO_SOURCE')
    SHOW_IMAGE = os.getenv(key='SHOW_IMAGE')
    SAVE_VIDEO = os.getenv(key='SAVE_VIDEO')

    # calling generator that yields a list with info about detected objects
    for list_obj_info in detect(vid_path=VIDEO_SOURCE, show_image=SHOW_IMAGE, save_video=SAVE_VIDEO):

        # print info about objects
        for obj_info in list_obj_info:
            print(obj_info, "\n")