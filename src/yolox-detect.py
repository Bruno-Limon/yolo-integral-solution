from collections import defaultdict
import numpy as np
import os
import gc

import torch
import cv2
from dotenv import load_dotenv

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

    while cap.isOpened():
        success, frame = cap.read()
        # frame_counter += fps/2 # every value corresponding to the vid's fps advances the frame by 1 sec
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

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
            yield frame_info_dict

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

    # del model_obj
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # load environment variables
    load_dotenv('.env')
    VIDEO_SOURCE = os.getenv(key='VIDEO_SOURCE')
    SHOW_IMAGE = os.getenv(key='SHOW_IMAGE')
    SAVE_VIDEO = os.getenv(key='SAVE_VIDEO')

    # calling generator that yields a json object with info about each frame and the objects in it
    for frame_info in detect(vid_path=VIDEO_SOURCE, show_image=True, save_video=SAVE_VIDEO):
        print(frame_info, "\n")