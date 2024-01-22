import cv2
import numpy as np

# global flag to evaluate if environment variables are True
ENV_VAR_TRUE_LABEL = "true"

def euclid_distance(pt1, pt2):
    """
    TODO
    """
    dist = np.sqrt(np.sum(np.square(np.array(pt1) - np.array(pt2))))
    return dist

# class to instantiate every object detected and its attributes,
# in case of car or bicycle set "keypoints" to 0
class DetectedObject:
    def __init__(self, id:int, label_num:int, label_str:str, conf:float, bbox_xy:list, bbox_wh:list)->None:
        """
        class of objects detected by the object detection algorithm

        attributes:
            __init__ id: id assigned to object, useful only with tracking
            __init__ label_num: number of object category
            __init__ label_str: label of object category
            __init__ conf: confidence of the detection
            __init__ bbox_xy: bounding box corners coordinates [x1,y1,x2,y2]
            __init__ bbox_wh: bounding box center and width/height [x,y,w,h]
            TODO
            is_down: bool variable telling if person is on the ground
            is_in_zone: bool variable telling if object is inside zone
            time_in_zone: amount of time inside the zone
            info: empty dict to fill with object's information
        """
        self.id = id
        self.label_num = label_num
        self.label_str = label_str
        self.conf = conf
        self.bbox_xy = bbox_xy # left upper corner and right lower corner
        self.bbox_wh = bbox_wh # center of bbox together with width and height
        self.track = []
        self.is_moving = False
        self.direction = "-"
        self.speed = 0
        self.is_down = False
        self.is_in_zone = False
        self.time_in_scene = 0
        self.time_in_zone = 0
        self.info = {}

        self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2 = self.bbox_xy
        self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h = self.bbox_wh

    # draw each bounding box with the color selected in "labels_dict", writes id and confidence of detected object
    def draw_boxes(self, frame, labels_dict:dict)->None:
        """
        draw each bounding box with the color provided in "labels_dict", uses self.bbox_xy and self.bbox_wh

        params:
            frame: current frame given by opencv
            labels_dict: dictionary with corresponding labels for categories of objects
        """

        text = f"{self.id} {self.conf}"
        txt_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=1)[0]

        # bbox rectangle
        cv2.rectangle(img=frame,
                    pt1=(self.bbox_x1,self.bbox_y1),
                    pt2=(self.bbox_x2,self.bbox_y2),
                    color=labels_dict[int(self.label_num)][1],
                    thickness=2)

        # header of bbox to put info into
        cv2.rectangle(img=frame,
                    pt1=(self.bbox_x1 - 1, self.bbox_y1 - int(1.5 * txt_size[1])),
                    pt2=(self.bbox_x1 + txt_size[0] + 2, self.bbox_y1),
                    color=labels_dict[int(self.label_num)][1],
                    thickness=-1)

        # information about object, id and conf
        cv2.putText(img=frame,
                    text=text,
                    org=(self.bbox_x1, self.bbox_y1 - int(.5 * txt_size[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(255,255,255),
                    thickness=1)

        # central lower point of bounding box, used for debugging
        if self.label_num == 0:
            cv2.circle(img=frame, center=(self.bbox_x, self.bbox_y+int((self.bbox_h/2))), radius=4, color=(0,255,0), thickness=-1)

    def get_is_down(self, cap, frame, show_man_down:str)->None:
        """
        detect man down, using the width and height of bbox_wh, if w > h, then man_down = True

        params:
            cap: video capture
            frame: current frame given by opencv
            show_man_down: boolean variable to show man_down in visualization, from env_vars
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        no_zone = height/15
        polygon = [[0,no_zone], [width,no_zone], [width,height-no_zone], [0,height-no_zone]]

        # zone to count people in
        zone_poly = np.array([polygon[0], # left upper corner
                              polygon[1], # right upper corner
                              polygon[2], # right lower corner
                              polygon[3]], np.int32) # left lower corner
        zone_poly = zone_poly.reshape((-1, 1, 2))
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(self.bbox_x, self.bbox_y+int((self.bbox_h/2))), measureDist=False)
        # cv2.polylines(img=frame, pts=[zone_poly], isClosed=True, color=(255,0,0), thickness=2)

        self.bbox_quotient = self.bbox_w / self.bbox_h
        if self.bbox_quotient >= 1.5 and self.label_num == 0:
            self.is_down = True
        elif self.bbox_quotient >= 1 and self.bbox_quotient < 1.5 and self.label_num == 0:
            if point_in_polygon == 1.0 or point_in_polygon == 0.0:
                self.is_down = True

        if show_man_down == ENV_VAR_TRUE_LABEL and self.is_down == True:
            info_text = f"{self.id} {self.conf}"
            info_size = cv2.getTextSize(text=info_text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=1)[0]

            cv2.rectangle(img=frame,
                          pt1=(self.bbox_x1,self.bbox_y1),
                          pt2=(self.bbox_x2,self.bbox_y2),
                          color=(0,0,255),
                          thickness=3)

            cv2.rectangle(img=frame,
                          pt1=(self.bbox_x1 - 1, self.bbox_y1 - int(1.5 * info_size[1])),
                          pt2=(self.bbox_x1 + info_size[0] + 2, self.bbox_y1),
                          color=(0,0,255),
                          thickness=-1)

            cv2.putText(img=frame,
                        text=info_text,
                        org=(self.bbox_x1, self.bbox_y1 - int(.5 * info_size[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5,
                        color=(255,255,255),
                        thickness=1)

        #     cv2.rectangle(img=frame, pt1=(self.bbox_x1-2, self.bbox_y1+(self.bbox_h-50)), pt2=(self.bbox_x2+2, self.bbox_y2),
        #                   color=(0,0,0), thickness=-1)
        #     cv2.putText(img=frame, text="FALLEN", org=(self.bbox_x1, self.bbox_y2-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=1, color=(255,255,255), thickness=1)

    def build_track(self, init_vars):
        """
        TODO
        """
        self.track = init_vars['track_history_dict'][self.id]
        self.track.append((self.bbox_x, self.bbox_y))

        if len(self.track) > 45:  # how many frames(seconds) to keep the track
            self.track.pop(0)

        if self.track:
            self.track_first = self.track[0]
            self.track_last = self.track[-1:][0]
            self.track_distance = euclid_distance(self.track_first, self.track_last)

             # track moves more than x pixels, depending on pixel per meter
            if self.track_distance > 10:
                self.is_moving = True

    def draw_tracks(self, frame)->None:
        """
        TODO draws a line following the movement of detected objects during a certain amount of frames

        params:
          frame: current frame given by opencv
        """
        color = (255, 120, 255, .5)

        if self.track:
            # draw the tracking line position by position
            points = np.hstack(self.track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(img=frame, pts=[points], isClosed=False,
                          color=color, thickness=3)

            # turns tracking history into vector
            # cv2.arrowedLine(img=frame, pt1=self.track_first, pt2=self.track_last, color=color, thickness=2)

    def estimate_direction(self, frame, labels_dict:dict)->None:
        """
        TODO
        """
        if self.track and self.is_moving:
            x1, y1 = self.track_first
            x2, y2 = self.track_last

            new_x = x2 - x1
            new_y = y1 - y2 # invert y-axis due to opencv inversion

            theta = [np.arctan2((new_y,y1),(new_x,x1))][0][0]
            angle = theta * 180 / np.pi
            if angle < 0:
                angle = 360 + angle

            if angle >= 330 or (angle >= 0  and angle < 30):
                self.direction = "E"
            elif angle >= 30 and angle < 60:
                self.direction = "NE"
            elif angle >= 60 and angle < 120:
                self.direction = "N"
            elif angle >= 120 and angle < 150:
                self.direction = "NW"
            elif angle >= 150 and angle < 210:
                self.direction = "W"
            elif angle >= 210 and angle < 240:
                self.direction = "SW"
            elif angle >= 240 and angle < 300:
                self.direction = "S"
            elif angle >= 300 and angle < 330:
                self.direction = "SE"

        text = self.direction
        txt_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.35, thickness=1)[0]

        if self.is_down == False:
            cv2.rectangle(img=frame,
                        pt1=(self.bbox_x1 - 1, self.bbox_y2 - int(2.1*txt_size[1])),
                        pt2=(self.bbox_x1 + int(txt_size[0]*1.5), self.bbox_y2),
                        color=labels_dict[int(self.label_num)][1],
                        thickness=-1)

            cv2.putText(img=frame,
                        text=text,
                        org=(self.bbox_x1, self.bbox_y2 - int(.7*txt_size[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5,
                        color=(255,255,255),
                        thickness=1)

    """TODO WORK IN PROGRESS TODO"""
    def estimate_speed(self, frame, fps, init_vars:dict)->None:
        """
        """
        # TO USE IN DETECTION_UTILS.PY:
        # for obj in list_objects:
        #     obj_is_in_zone = obj.get_is_in_zone(init_vars['cross_poly'])
        #     if obj_is_in_zone:
        #         init_vars['time_in_cross_dict'][obj.id] += 1
        #     # show crossing speed when object leaves crossing zone
        #     if not obj_is_in_zone and init_vars['time_in_cross_dict'][obj.id] > 0:
        #         obj.estimate_speed(frame, fps, init_vars)

        labels_dict = init_vars['labels_dict']
        time_in_cross_dict = init_vars['time_in_cross_dict']

        time = time_in_cross_dict[self.id] # access dict with id and amount of time in zone
        if fps > 0:
            time = time/fps
        else:
            time = 0
        self.speed = round(2.1/time, 1)

        text = f"{self.speed} km/h"
        txt_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4, thickness=1)[0]

        cv2.rectangle(img=frame,
                      pt1=(self.bbox_x1 - 1, self.bbox_y2 + int(1.5*txt_size[1])),
                      pt2=(self.bbox_x1 + int(txt_size[0]*1.3), self.bbox_y2),
                      color=labels_dict[int(self.label_num)][1],
                      thickness=-1)

        cv2.putText(img=frame,
                    text=text,
                    org=(self.bbox_x1, self.bbox_y2 + int(1.2 * txt_size[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(255,255,255),
                    thickness=1)

    def get_is_in_zone(self, zone_poly)->bool:
        """
        determines if person is inside the given zone or not

        params:
            zone_poly: np.array with coordinates for the zone of interest
        """
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(self.bbox_x, self.bbox_y+int((self.bbox_h/2))), measureDist=False)
        if point_in_polygon == 1.0 or point_in_polygon == 0.0:
            self.is_in_zone = True
            return True
        else:
            return False

    def set_time_in_scene(self, frame, fps, init_vars):
        """
        TODO
        """
        time_in_scene_dict = init_vars['time_in_scene_dict']

        time = time_in_scene_dict[self.id] # access dict with id and amount of time in zone
        if fps > 0:
            time = time/fps
        else:
            time = 0
        self.time_in_scene = time

    def set_time_in_zone(self, frame, fps:int, init_vars:dict, show_time_zone:str):
        """
        determines amount of time a person has spent inside the given area

        param:
            frame: current frame given by opencv
            fps: video's fps
            init_vars:
            show_time_zone: boolean variable to show time on top of object in visualization, from env_vars
        """
        zone_poly = init_vars['zone_poly']
        time_in_zone_dict = init_vars['time_in_zone_dict']

        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(self.bbox_x, self.bbox_y+int((self.bbox_h/2))), measureDist=False)
        time = time_in_zone_dict[self.id] # access dict with id and amount of time in zone
        if fps > 0:
            time = round(time/fps, 1)
        else:
             time = 0
        self.time_in_zone = time

        if show_time_zone == ENV_VAR_TRUE_LABEL and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            text = str(time)
            txt_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4, thickness=1)[0]
            cv2.rectangle(img=frame, pt1=(self.bbox_x1-2, self.bbox_y1-40),
                          pt2=(self.bbox_x1 + txt_size[0] + 4, self.bbox_y2-(self.bbox_h+20)),
                          color=(0,0,0), thickness=-1)
            cv2.putText(img=frame, text=text, org=(self.bbox_x1 + 2, self.bbox_y1 - 25),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4, color=(255,255,255), thickness=1)

    def obj_info(self)->dict:
        """
        puts object info into a dictionary to incorporate into the sending of messages
        about the detected objects in the video
        """
        self.info = {"id": self.id,
                     "class": self.label_str,
                     "confidence": self.conf,
                     "bbox_xywh": self.bbox_wh,
                     "is_down": self.is_down,
                     "is_moving": self.is_moving,
                     "direction": self.direction,
                     "is_in_zone": self.is_in_zone,
                     "time_in_zone": self.time_in_zone,
        }

        return self.info

    # count number of people entering and leaving
    def enter_leave(self, frame, frame_width, door_poly, door2_poly, init_vars):
        """
        TODO

        params:
            frame:
            frame_width:
            door_poly:
            door2_poly:
            init_vars:
        """
        # lower point of bounding box, representing the feet
        feet_bb = (self.bbox_x, self.bbox_y+int((self.bbox_h/2)))

        # cv2.polylines(img=frame, pts=[door_poly], isClosed=True, color=(0,204,255), thickness=2)
        # cv2.polylines(img=frame, pts=[door2_poly], isClosed=True, color=(0,204,255), thickness=2)

        # people entering
        # if feet of person enter 1st threshold, add them to dict of people entering
        point_in_polygon = cv2.pointPolygonTest(contour=door2_poly, pt=feet_bb, measureDist=False)
        if point_in_polygon >=0:
            init_vars['people_entering_dict'][self.id] = feet_bb

        # when they step on 2nd threshold, add them to the set
        if self.id in init_vars['people_entering_dict']:
            point_in_polygon2 = cv2.pointPolygonTest(contour=door_poly, pt=feet_bb, measureDist=False)
            if point_in_polygon2 >=0:
                init_vars['entering'].add(self.id)

        # people leaving
        # same logic as before but reversing the order of the thresholds
        point_in_polygon3 = cv2.pointPolygonTest(contour=door_poly, pt=feet_bb, measureDist=False)
        if point_in_polygon3 >=0:
            init_vars['people_leaving_dict'][self.id] = feet_bb
        if self.id in init_vars['people_leaving_dict']:
            point_in_polygon4 = cv2.pointPolygonTest(contour=door2_poly, pt=feet_bb, measureDist=False)
            if point_in_polygon4 >=0:
                init_vars['leaving'].add(self.id)

        cv2.rectangle(img=frame, pt1=(frame_width-110,0), pt2=(frame_width,55), color=(0,0,0), thickness=-1)
        cv2.putText(img=frame, text=f"Entering: {len(init_vars['entering'])}", org=(frame_width-110, 25),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255,255,255), thickness=1)
        cv2.putText(img=frame, text=f"Leaving: {len(init_vars['leaving'])}", org=(frame_width-110, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255,255,255), thickness=1)
