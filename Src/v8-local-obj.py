import cv2
import math
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# enable rtsp capture for opencv
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# text settings
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2

# mapping of keypoints with their respective position in the array returned by "results.keypoints"
BODY_MAP = {"nose": 0, "eye_sx": 1, "eye_dx": 2, "ear_sx": 3,
            "ear_dx": 4, "shoulder_sx": 5, "shoulder_dx": 6,
            "elbow_sx": 7, "elbow_dx": 8, "wrist_sx": 9,
            "wrist_dx": 10, "hip_sx": 11, "hip_dx": 12,
            "knee_sx": 13, "knee_dx": 14, "foot_sx": 15, "foot_dx": 16
}

# empty dictionary to populate using x,y coordinates of keypoints
kp_position_dict = {"nose": [], "eye_sx": [], "eye_dx": [],
                   "ear_sx": [], "ear_dx": [], "shoulder_sx": [],
                   "shoulder_dx": [], "elbow_sx": [], "elbow_dx": [],
                   "wrist_sx": [], "wrist_dx": [], "hip_sx": [],
                   "hip_dx": [], "knee_sx": [], "knee_dx": [],
                   "foot_sx": [], "foot_dx": []
}

# dictionary to map the class number obtained with yolo with its name and color for bounding boxes
labels_dict = {0: ["person", (209,209,0)],
               1: ["bicycle", (47,139,237)],
               2: ["car", (42,237,139)]
}

###################################################################################

# class to instantiate every object detected and its attributes,
# in case of car or bicycle set "keypoints" to 0
class DetectedObject:
    def __init__(self, id, label_num, label_str, conf, bbox_xy, bbox_wh, keypoints):
        self.id = id
        self.label_num = label_num
        self.label_str = label_str
        self.conf = conf
        self.bbox_xy = bbox_xy # left upper corner and right lower corner
        self.bbox_wh = bbox_wh # center of bbox together with width and height
        self.keypoints = keypoints
        self.is_down = False
        self.info = {}

    # draw each bounding box with the color selected in "labels_dict", writes id and confidence of detected object
    def draw_boxes(self, frame):
        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        cv2.rectangle(img=frame, pt1=(x1,y1), pt2=(x2,y2),
                      color=labels_dict[int(self.label_num)][1], thickness=thickness)
        cv2.rectangle(img=frame, pt1=(x1-thickness, y1-20), pt2=(x2+thickness, y2-h),
                      color=labels_dict[int(self.label_num)][1], thickness=-1)

        # information about each object, can contain any class attribute
        cv2.putText(frame, f"id:{self.id} {self.conf}", org=(x1, y1-5), fontFace=font,
                    fontScale=.5, color=(255,255,255), thickness=1)

    # detects if someone is on the ground by using the keypoints of each object and measuring the angle of the
    # person's body with respect to the ground
    def detect_is_down(self, frame, show_keypoints, show_down_onscreen):
        if self.label_str == "person":
            # i is the number of the person, j is the body part
            for i in range(len(self.keypoints)):
                feet_xy = [int((self.keypoints[i][16][0] + self.keypoints[i][15][0])/2),
                           int((self.keypoints[i][16][1] + self.keypoints[i][15][1])/2)]

                # draw keypoints on the screen
                if show_keypoints:
                    for j in range(17):
                        cv2.circle(img=frame, center=(self.keypoints[i][j][0], self.keypoints[i][j][1]), radius=2,
                                   color=(0,255,0), thickness=thickness)
                        cv2.line(img=frame, pt1=(self.keypoints[i][0][0], self.keypoints[i][0][1]),
                                 pt2=(feet_xy[0], feet_xy[1]), color=(255,0,255), thickness=thickness)

            # angle with respect to the ground
            angle = math.degrees(math.atan2(feet_xy[1] - self.keypoints[i][0][1],
                                            feet_xy[0] - self.keypoints[i][0][0]))

            # the angle determines if the person is on the ground or not
            if ((angle < 50) or (angle > 150)):
                self.is_down = True
                x, y, w, h = self.bbox_wh
                x1, y1, x2, y2 = self.bbox_xy
                cv2.rectangle(img=frame, pt1=(x1-thickness, y1-50), pt2=(x2+thickness, y2-(h+20)),
                            color=(0,0,0), thickness=-1)
                cv2.putText(frame, "FALLEN", org=(x1, y1-25), fontFace=font,
                            fontScale=1, color=(255,255,255), thickness=1)

    # prints or returns all info about the detected objects in each frame
    def obj_info(self):
        #for key in BODY_MAP:
        #    kp_position_dict[key] = [self.keypoints[0][BODY_MAP[key]][0], self.keypoints[0][BODY_MAP[key]][1]]

        self.info = {"date_time": datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                    "id": self.id,
                    "class": self.label_str,
                    "confidence": self.conf,
                    "bbox_xywh": self.bbox_wh,
                    "keypoints": 0,
                    "is_down": self.is_down
        }

        print(self.info, "\n")

# dynamically creating objects from yolo detection and pose estimation
def generate_objects(results_pose, results_obj):
    list_objects = []
    # pose detection results
    for r in results_pose:
        kp = r.keypoints.xy.numpy().astype(np.int32).tolist()

    # object detection results
    for r in results_obj:
        boxes = r.boxes.numpy()

        # only creating an object if it belongs to the chosen classes
        for box in (x for x in boxes if x.cls in [0,1,2]):
            if box.cls == 0: # == person
                keypoints = kp
            else:
                # if object is not a person, keypoints = 0
                keypoints = 0
            if box.id != None:
                list_objects.append(DetectedObject(id=int(box.id),
                                                   label_num=int(box.cls),
                                                   label_str=labels_dict[int(box.cls)][0],
                                                   conf=round(float(box.conf),2),
                                                   bbox_xy=box.xyxy[0].astype(np.int32).tolist(),
                                                   bbox_wh=box.xywh[0].astype(np.int32).tolist(),
                                                   keypoints=keypoints
                                                   )
                                    )
    return list_objects

# one or more zones in which to count the people inside
def count_zone(frame, list_objects, poly, show_zone_onscreen):
    count_people_polygon = 0
    for obj in (obj for obj in list_objects if obj.label_num == 0):
        x,y,w,h = obj.bbox_wh
        point_in_polygon = cv2.pointPolygonTest(contour=poly, pt=(x, y+int((h/2))), measureDist=False)
        if point_in_polygon == 1.0 or point_in_polygon == 0.0: count_people_polygon += 1

    if show_zone_onscreen:
        cv2.polylines(img=frame, pts=[poly], isClosed=True, color=(255,0,0), thickness=thickness)
        cv2.rectangle(img=frame, pt1=(850, 670), pt2=(900, 710), color=(255,0,0), thickness=-1)
        cv2.putText(img=frame, text=str(count_people_polygon), org=(865, 700), fontFace=font,
                    fontScale=1, color=(255,255,255), thickness=2)

# counting overall objects on screen, including people, bikes and cars
def count_objs(frame, list_objects, show_count_onscreen):
    count_people = sum(1 for obj in list_objects if obj.label_num == 0)
    count_bike = sum(1 for obj in list_objects if obj.label_num == 1)
    count_car = sum(1 for obj in list_objects if obj.label_num == 2)

    if show_count_onscreen:
        cv2.rectangle(img=frame, pt1=(0,0), pt2=(110,80), color=(0,0,0), thickness=-1)
        cv2.putText(img=frame, text=f"People: {count_people}", org=(0, 25), fontFace=font,
                    fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Bicycles: {count_bike}", org=(0, 50), fontFace=font,
                    fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Cars: {count_car}", org=(0, 75), fontFace=font,
                    fontScale=.6, color=(255,255,255), thickness=1)

###################################################################################

# feed the video soruce and apply yolo models, then call the chosen functions for the different tasks as needed
def detect(vid_path, zone_poly, do_man_down, show_keypoints, show_down_onscreen, do_count_objs,
           show_count_onscreen, show_box, do_count_zone, show_zone_onscreen, print_obj_info, save_video):
    model_pose = YOLO("yolov8n-pose.pt")
    model_obj = YOLO("yolov8n.pt")

    frame_counter = 0
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_video:
        output = cv2.VideoWriter("output-demo.avi", cv2.VideoWriter_fourcc(*'MPEG'),
                                 fps=fps, frameSize=(width, height))

    while cap.isOpened():
        success, frame = cap.read()
        frame_counter += fps # this advances 1 seconds between each frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        if success:
            results_pose = model_pose.predict(frame, save=False, stream=True, verbose=False, conf=.40)
            results_obj = model_obj.track(frame, save=False, stream=True, verbose=False, conf=.40, tracker="bytetrack.yaml")

            list_objects = generate_objects(results_pose, results_obj)

            if show_box: [obj.draw_boxes(frame) for obj in list_objects]
            if do_man_down: [obj.detect_is_down(frame, show_keypoints, show_down_onscreen) for obj in list_objects]
            if do_count_objs: count_objs(frame, list_objects, show_count_onscreen)
            if print_obj_info: [obj.obj_info() for obj in list_objects]
            if do_count_zone: count_zone(frame, list_objects, zone_poly, show_zone_onscreen)

            # write output video
            if save_video: output.write(frame)

            # display the annotated frame
            cv2.imshow("Demo", frame)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # break the loop if the end of the video is reached (comment out when using stream)
        else:
            break

    # release the video capture object and close the display window
    cap.release()
    if save_video: output.release()
    cv2.destroyAllWindows()

###################################################################################

if __name__ == "__main__":

    # video source
    vid_path = '../Data/vid5.mp4'

    # zone to count people in
    zone_poly = np.array([[460, 570], #x1, y1 = left upper corner
                          [1270, 570],#x2, y2 = right upper corner
                          [1265, 710],#x3, y3 = right lower corner
                          [430, 710]], np.int32) #x4, y4 = left lower corner
    zone_poly = zone_poly.reshape((-1, 1, 2))

    # calling main detection function, passing all necessary arguments
    detect(vid_path=vid_path,
           zone_poly=zone_poly,
           do_man_down=False,
           show_keypoints=True,
           show_down_onscreen=True,
           do_count_objs=True,
           show_count_onscreen=True,
           show_box=True,
           do_count_zone=True,
           show_zone_onscreen=True,
           print_obj_info=False,
           save_video=False)