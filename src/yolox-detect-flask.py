from collections import defaultdict
import numpy as np
import os
import gc

import torch
import cv2
import time

from utils import *
from Detected_object import DetectedObject
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


if __name__ == "__main__":

    from frame_singleton import current_frame
    from Flask_frame import Flask_frame
    from flask import Flask, request, Response
    import threading

    app = Flask(__name__)
    Flask_frame = Flask_frame(app)

    # load environment variables
    VIDEO_SOURCE = 'https://nvidia.box.com/shared/static/veuuimq6pwvd62p9fresqhrrmfqz0e2f.mp4' #'rtsp://admin:T0lstenc088@abyss88.ignorelist.com/1' #os.getenv(key='VIDEO_SOURCE')
    SHOW_IMAGE = False #os.getenv(key='SHOW_IMAGE')
    SAVE_VIDEO = False #os.getenv(key='SAVE_VIDEO')
    EXPOSE_STREAM = True #os.getenv(key='EXPOSE_STREAM')
    RUN_WAIT_TIME = 100 #int(os.getenv(key='RUN_WAIT_TIME'))

    if EXPOSE_STREAM:
        server_thread = threading.Thread(target=Flask_frame.run_server)
        server_thread.daemon = True
        server_thread.start()

    # calling generator that yields a json object with info about each frame and the objects in it
    import asyncio

    # Video streaming page
    @app.route('/video')
    def video():
        # Simple login
        login = request.args.get('login')
        # Hardcoded password to login
        if login == 'simple_access1':
            # If success return the stream
            return Response(Flask_frame.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return Response("Invalid login",mimetype='text/plain')

    # Homepage
    @app.route('/')
    def index():
        return "Service is up!"

    async def loop_main():
        global current_frame
        while True:
            for frame_info, frame in detect(vid_path=VIDEO_SOURCE, show_image=SHOW_IMAGE, save_video=SAVE_VIDEO):
                print(frame_info, "\n")

                # with frame_lock:
                current_frame = frame

    loop = asyncio.new_event_loop()
    loop.run_until_complete(loop_main())

    #'rtsp://admin:T0lstenc088@abyss88.ignorelist.com/1'