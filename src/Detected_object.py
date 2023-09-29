import cv2
import numpy as np

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
    def draw_boxes(self, frame, labels_dict)->None:
        """Raw each bounding box with the color selected in "labels_dict",
         writes id and confidence of detected object.

        :param frame:
        :param ciccio:

        :return: None
        """

        x1, y1, x2, y2 = self.bbox_xy
        x, y, w, h = self.bbox_wh

        text = f"id:{self.id} {self.conf}"
        txt_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=1)[0]

        # bbox rectangle
        cv2.rectangle(img=frame,
                    pt1=(x1,y1),
                    pt2=(x2,y2),
                    color=labels_dict[int(self.label_num)][1],
                    thickness=2)

        # header of bbox to put info into
        cv2.rectangle(img=frame,
                    pt1=(x1 - 2, y1 - int(1.5 * txt_size[1])),
                    pt2=(x1 + txt_size[0] + 2, y1),
                    color=labels_dict[int(self.label_num)][1],
                    thickness=-1)

        # information about object, id and conf
        cv2.putText(img=frame,
                    text=text,
                    org=(x1, y1 - int(.5 * txt_size[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(255,255,255),
                    thickness=1)

        # central lower point of bounding box
        if self.label_num == 0:
            cv2.circle(img=frame, center=(x, y+int((h/2))), radius=4, color=(0,255,0), thickness=-1)

    # detect man down, using the width and height of bounding box, if w > h, then man_down
    def get_is_down(self, frame, show_man_down):
        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        # the angle determines if the person is on the ground or not
        if w > h and self.label_num == 0:
            self.is_down = True

        if show_man_down == 'True':
            cv2.rectangle(img=frame, pt1=(x1-2, y1+(h-50)), pt2=(x2+2, y2),
                          color=(0,0,0), thickness=-1)
            cv2.putText(img=frame, text="FALLEN", org=(x1, y2-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255,255,255), thickness=1)

    # # draw a line following the movement of detected people during a certain amount of frames
    # def draw_tracks(self, frame, track_history_dict):
    #     if self.label_num == 0:
    #         x, y, w, h = self.bbox_wh
    #         track = track_history_dict[self.id]
    #         track.append((float(x), float(y)))  # x, y center point
    #         if len(track) > 60:  # how many frames to keep the track
    #             track.pop(0)

    #         # draw the tracking lines
    #         points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
    #         cv2.polylines(img=frame, pts=[points], isClosed=False, color=(230,53,230), thickness=5)

    # determines if person is inside a defined area or not
    def get_is_in_zone(self, zone_poly):
        x, y, w, h = self.bbox_wh

        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(x, y+int((h/2))), measureDist=False)
        if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            self.is_in_zone = True

        return self.id, self.is_in_zone

    # shows amount of time a person has spent inside area
    def draw_time_zone(self, frame, time_in_zone_dict, fps, zone_poly, show_time_zone):
        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        point_in_polygon = cv2.pointPolygonTest(contour=zone_poly, pt=(x, y+int((h/2))), measureDist=False)
        if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
            time = time_in_zone_dict[self.id] # access dict with id and amount of time in zone
            time = round(time/fps, 1)
            self.time_in_zone = time

            if show_time_zone == 'True':
                cv2.rectangle(img=frame, pt1=(x1-2, y1-40), pt2=(x2+2, y2-(h+20)),
                            color=(0,0,0), thickness=-1)
                cv2.putText(img=frame, text=str(time), org=(x1+int((w/3)), y1-25),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255,255,255), thickness=2)

#     # count number of people entering and leaving
#     def enter_leave(self, frame, frame_width, door_poly, door2_poly):
#         x, y, w, h = self.bbox_wh

#         # lower point of bounding box, representing the feet
#         feet_bb = (x, y+int((h/2)))

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

    # prints or returns all info about the detected objects in each frame
    def obj_info(self):
        self.info = {"id": self.id,
                     "class": self.label_str,
                     "confidence": self.conf,
                     "bbox_xywh": self.bbox_wh,
                     "is_down": self.is_down,
                     "is_in_zone": self.is_in_zone,
                     "time_in_zone": self.time_in_zone,
        }

        return self.info
