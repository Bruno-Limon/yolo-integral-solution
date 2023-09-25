from collections import defaultdict
import numpy as np
import os
import gc

import torch
import cv2
import time
from utils import *
from YOLOX.yolox.exp import get_exp
from YOLOX.tools.detect import process_frame


# enable rtsp capture for opencv
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# dictionary to map the class number obtained with yolo with its name and color for bounding boxes
labels_dict = get_labels_dict()
# zone to count people in
zone_poly = get_zone_poly()

# NEEDS TRACKING
# # dictionaries containing id of people entering/leaving and sets to count them
people_entering_dict = {}
entering = set()
people_leaving_dict = {}
leaving = set()

# door thresholds
door_poly, door2_poly = get_door_thresholds()

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
    def get_is_down(self, frame):
        x, y, w, h = self.bbox_wh
        x1, y1, x2, y2 = self.bbox_xy

        # the angle determines if the person is on the ground or not
        if w > h and self.label_num == 0:
            self.is_down = True
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

    # # determines if person is inside a defined area or not
    # def get_is_in_zone(self, poly):

    #     x, y, w, h = self.bbox_wh

    #     point_in_polygon = cv2.pointPolygonTest(contour=poly, pt=(x, y+int((h/2))), measureDist=False)
    #     if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
    #         self.is_in_zone = True

    #     return self.id, self.is_in_zone

#     # shows amount of time a person has spent inside area
#     def draw_time_zone(self, frame, poly, time_in_zone_dict, fps):

#         x, y, w, h = self.bbox_wh
#         x1, y1, x2, y2 = self.bbox_xy

#         point_in_polygon = cv2.pointPolygonTest(contour=poly, pt=(x, y+int((h/2))), measureDist=False)
#         if self.label_num == 0 and (point_in_polygon == 1.0 or point_in_polygon == 0.0):
#             time = time_in_zone_dict[self.id] # access dict with id and amount of time in zone
#             time = round(time/fps, 1)
#             cv2.rectangle(img=frame, pt1=(x1-thickness, y1-40), pt2=(x2+thickness, y2-(h+20)),
#                             color=(0,0,0), thickness=-1)
#             cv2.putText(img=frame, text=str(time), org=(x1+int((w/3)), y1-25), fontFace=font,
#                                 fontScale=.5, color=(255,255,255), thickness=2)

#             self.time_in_zone = time

#     # count number of people entering and leaving
#     def enter_leave(self, frame, frame_width):

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


def detect(vid_path, show_image, save_video):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    is_stream = True #type(vid_path) is int or vid_path.startswith('rtsp')

    LOG_KW = "LOG"
    print(f"{LOG_KW}: model loaded, starting detection")

    # NEEDS TRACKING
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

    # Initialize frame skipping mechanism
    frames_to_skip=1
    # Read the whole input
    while cap.isOpened():
        print('Connection established')
        # Custom frames skipping
        # Opencv cap.set takes 1s
        start = time.time()
        if not is_stream:
            while frames_to_skip>0:
                frames_to_skip-=1
                success, frame = cap.read()
                continue
        else:
            success, frame = cap.read()

        # If failed and stream reconnect
        if not success and is_stream:
            cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
            print('Reconnecting to the stream')
            continue
        # If failed and not stream -> video finished
        elif not success:
            break

        # If frame is read, compute outputs
        if success:
            print(f"{LOG_KW}: video detected")
            model = get_exp(exp_file=None, exp_name="yolox-nano")
            results_image = process_frame(model_name="yolox_nano.pth", exp=model, frame=frame)

            list_objects = generate_objects(DetectedObject, results_image, labels_dict)

            # show bounding boxes
            [obj.draw_boxes(frame) for obj in list_objects]

            # NEEDS TRACKING
            # draw tracks
            # [obj.draw_tracks(frame, track_history_dict) for obj in list_objects]

            # detect man down
            [obj.get_is_down(frame) for obj in list_objects]

            # NEEDS TRACKING
            # # for every person inside an area, count the number of frames
            # for obj in list_objects:
            #     obj_id, obj_is_in_zone, = obj.get_is_in_zone(zone_poly)
            #     if obj_is_in_zone:
            #         time_in_zone_dict[obj_id] += 1

            # count objects
            number_objs = count_objs(frame, list_objects)

            # count people in zone
            number_people_zone = count_zone(frame, list_objects, zone_poly)

            # NEEDS TRACKING
            # # count people entering and leaving a certain area
            # [obj.enter_leave(frame, width) for obj in list_objects]

            # NEEDS TRACKING
            # # show time inside zone on top of people's boxes
            # [obj.draw_time_zone(frame, zone_poly, time_in_zone_dict, fps) for obj in list_objects]

            # get object info
            obj_info = []
            for obj in list_objects:
                x = obj.obj_info()
                obj_info.append(x)

            # get frame info
            print(f"{LOG_KW}: results:", "\n")
            frame_info_dict = send_frame_info(number_objs, number_people_zone, cap, obj_info)
            yield frame_info_dict, frame

            # write output video
            if save_video == True:
                output.write(frame)

            # display the annotated frame
            if show_image == True:
                cv2.imshow("Demo", frame)

            # break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        end = time.time()
        elapsed = (end-start)
        frames_to_skip=int(fps*elapsed)

        # Left here for learning purposes. This is taking 1s
        # s2 = time.time()
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        # s3 = time.time()
        # print(f'opencv skip took {s3-s2} seconds')

    # release the video capture object and close the display window
    cap.release()

    if save_video == True:
        output.release()

    cv2.destroyAllWindows()

    # del model_obj
    torch.cuda.empty_cache()
    gc.collect()



###


from flask import Flask, Response, request
import threading

app = Flask(__name__)

# Global variables
# frame_lock used for race conditions on the global current_frame
# frame_lock = threading.Lock()
# current_frame produced and consumed

# Function to generate video frames from the stream
def generate_frames():
    global current_frame

    try:
        # Continuously
        while True:
            # Access to the current frame safely
            # with frame_lock:
            frame = current_frame

            # If exists, yield the frame
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    except GeneratorExit:
        # This is raised when the client disconnects
        print("Client disconnected")


# Video streaming page
@app.route('/video')
def video():
    # Simple login
    login = request.args.get('login')
    # Hardcoded password to login
    if login == 'simple_access1':
        # If success return the stream
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("Invalid login",mimetype='text/plain')

# Homepage
@app.route('/')
def index():
    return "Service is up!"


def run_server():
    app.run(host='0.0.0.0', port=8080, debug=False)


async def loop_main():
    global current_frame
    while True:
        for frame_info, frame in detect(vid_path=VIDEO_SOURCE, show_image=SHOW_IMAGE, save_video=SAVE_VIDEO):
            print(frame_info, "\n")

        # with frame_lock:
            current_frame = frame
###



if __name__ == "__main__":

    from frame_singleton import current_frame

    # load environment variables
    VIDEO_SOURCE = 'http://185.137.146.14:80/mjpg/video.mjpg' #'rtsp://admin:T0lstenc088@abyss88.ignorelist.com/1' #os.getenv(key='VIDEO_SOURCE')
    SHOW_IMAGE = False #os.getenv(key='SHOW_IMAGE')
    SAVE_VIDEO = False #os.getenv(key='SAVE_VIDEO')
    EXPOSE_STREAM = True #os.getenv(key='EXPOSE_STREAM')
    RUN_WAIT_TIME = 100 #int(os.getenv(key='RUN_WAIT_TIME'))

    if EXPOSE_STREAM:
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

    # calling generator that yields a json object with info about each frame and the objects in it
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(loop_main())

    #'rtsp://admin:T0lstenc088@abyss88.ignorelist.com/1'