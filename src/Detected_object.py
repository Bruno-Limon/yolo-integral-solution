import cv2
import numpy as np

# global flag to evaluate if environment variables are True
ENV_VAR_TRUE_LABEL = "true"

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
        self.is_down = False
        self.is_in_zone = False
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
                    pt1=(self.bbox_x1 - 2, self.bbox_y1 - int(1.5 * txt_size[1])),
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

    def get_is_down(self, frame, show_man_down:str)->None:
        """
        detect man down, using the width and height of bbox_wh, if w > h, then man_down = True

        params:
            frame: current frame given by opencv
            show_man_down: boolean variable to show man_down in visualization, from env_vars
        """

        if self.bbox_w > self.bbox_h and self.label_num == 0:
            self.is_down = True

        if show_man_down == ENV_VAR_TRUE_LABEL and self.is_down == True:
            cv2.rectangle(img=frame, pt1=(self.bbox_x1-2, self.bbox_y1+(self.bbox_h-50)), pt2=(self.bbox_x2+2, self.bbox_y2),
                          color=(0,0,0), thickness=-1)
            cv2.putText(img=frame, text="FALLEN", org=(self.bbox_x1, self.bbox_y2-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255,255,255), thickness=1)

    # def draw_tracks(self, frame, track_history_dict)->None:
    #     """
    #     draws a line following the movement of detected people during a certain amount of frames
    #
    #     params:
    #       frame: current frame given by opencv
    #       track_history_dict: dict of history of positions related to an id
    #     """
    #     if self.label_num == 0:
    #         track = track_history_dict[self.id]
    #         track.append((float(self.bbox_x), float(self.bbox_y)))
    #         if len(track) > 60:  # how many frames to keep the track
    #             track.pop(0)

    #         # draw the tracking lines
    #         points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
    #         cv2.polylines(img=frame, pts=[points], isClosed=False, color=(230,53,230), thickness=5)

    def get_is_in_zone(self, zone_poly)->tuple[int, bool]:
        """
        determines if person is inside the given zone or not

        params:
            zone_poly: np.array with coordinates for the zone of interest
        """
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(self.bbox_x, self.bbox_y+int((self.bbox_h/2))), measureDist=False)
        if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            self.is_in_zone = True

        return self.id, self.is_in_zone

    def draw_time_zone(self, frame, time_in_zone_dict:dict, fps:int, zone_poly, show_time_zone:str):
        """
        determines amount of time a person has spent inside the given area

        param:
            frame: current frame given by opencv
            time_in_zone_dict: dictionary with id and its amount of time inside zone
            fps: video's fps
            show_time_zone: boolean variable to show time on top of object in visualization, from env_vars
        """
        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(self.bbox_x, self.bbox_y+int((self.bbox_h/2))), measureDist=False)
        if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            time = time_in_zone_dict[self.id] # access dict with id and amount of time in zone
            time = round(time/fps, 1)
            self.time_in_zone = time

            if show_time_zone == ENV_VAR_TRUE_LABEL:
                cv2.rectangle(img=frame, pt1=(self.bbox_x1-2, self.bbox_y1-40), pt2=(self.bbox_x2+2, self.bbox_y2-(self.bbox_h+20)),
                            color=(0,0,0), thickness=-1)
                cv2.putText(img=frame, text=str(time), org=(self.bbox_x1+int((self.bbox_w/3)), self.bbox_y1-25),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255,255,255), thickness=2)

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
                     "is_in_zone": self.is_in_zone,
                     "time_in_zone": self.time_in_zone,
        }

        return self.info

#     # count number of people entering and leaving
#     def enter_leave(self, frame, frame_width, door_poly, door2_poly):
#         # lower point of bounding box, representing the feet
#         feet_bb = (self.bbox_x, self.bbox_y+int((self.bbox_h/2)))

#         # first threshold of door
#         cv2.polylines(img=frame, pts=[door_poly], isClosed=True, color=(0,204,255), thickness=thickness)
#         # second threshold
#         cv2.polylines(img=frame, pts=[door2_poly], isClosed=True, color=(0,204,255), thickness=thickness)

#         # people entering
#         # if feet of person enter 1st threshold, add them to dict of people entering
#         point_in_polygon = cv2.pointPolygonTest(contour=door2_poly, pt=feet_bb, measureDist=False)
#         if point_in_polygon >=0:
#             people_entering_dict[self.id] = feet_bb

#         # when they step on 2nd threshold, add them to the set
#         if self.id in people_entering_dict:
#             point_in_polygon2 = cv2.pointPolygonTest(contour=door_poly, pt=feet_bb, measureDist=False)
#             if point_in_polygon2 >=0:
#                 entering.add(self.id)

#         # people leaving
#         # same logic as before but reversing the order of the thresholds
#         point_in_polygon3 = cv2.pointPolygonTest(contour=door_poly, pt=feet_bb, measureDist=False)
#         if point_in_polygon3 >=0:
#             people_leaving_dict[self.id] = feet_bb
#         if self.id in people_leaving_dict:
#             point_in_polygon4 = cv2.pointPolygonTest(contour=door2_poly, pt=feet_bb, measureDist=False)
#             if point_in_polygon4 >=0:
#                 leaving.add(self.id)

#         cv2.rectangle(img=frame, pt1=(frame_width-110,0), pt2=(frame_width,55), color=(0,0,0), thickness=-1)
#         cv2.putText(img=frame, text=f"Entering: {len(entering)}", org=(frame_width-110, 25), fontFace=font,
#                 fontScale=.6, color=(255,255,255), thickness=1)
#         cv2.putText(img=frame, text=f"Leaving: {len(leaving)}", org=(frame_width-110, 50), fontFace=font,
#                 fontScale=.6, color=(255,255,255), thickness=1)
