from collections import defaultdict
import os
import gc
import sys
import time
import json
import logging
import torch
import cv2

from src.alert import Alert
from src.Detected_object import DetectedObject
from src.utils import *

import config
if config.env_vars_local is True:
    from dotenv import load_dotenv
    load_dotenv()

# yolo_x through super-gradients
if os.environ['LIBRARY'] == "supergradients":
    from src.super_gradients_detection import detect_sg

# yolo_x from source
if os.environ['LIBRARY'] == "yolox":
    sys.path.append(os.path.join(os.getcwd(), 'src', 'YOLOX'))
    from src.YOLOX.yolox.exp import get_exp
    from src.YOLOX.tools.detect import process_frame

# yolov8 through ultralytics
if os.environ['LIBRARY'] == "ultralytics":
    from ultralytics import YOLO


# enable rtsp capture for opencv
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENV_VAR_TRUE_LABEL = "true"


def set_initial_vars()->dict:
    """
    setting up necessary variables that need to be declared before the detection loop
    starts on each frame, so they are initialized at the beginning
    """
    print('LOG: clearing cuda')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    init_vars = {}
    # dictionary to map the class number obtained with yolo with its name and color for bbox
    init_vars['labels_dict'] = get_labels_dict()
    # zone to count people in
    init_vars['zone_poly'] = get_zone_poly(os.environ['ZONE_COORDS'])

    # NEEDS TRACKING TODO
    # dictionaries containing id of people entering/leaving and sets to count them
    # people_entering_dict = {}
    # entering = set()
    # people_leaving_dict = {}
    # leaving = set()

    # NEEDS TRACKING
    # store the track history
    init_vars['track_history_dict'] = defaultdict(lambda: [])
    # store the amount of frames spent inside zone
    init_vars['time_in_zone_dict'] = defaultdict(int)

    # initialize frame skipping mechanism
    #type(vid_path) is int or vid_path.startswith('rtsp')
    init_vars['is_stream'] = os.environ['IS_STREAM'] == ENV_VAR_TRUE_LABEL
    init_vars['frames_to_skip'] = 1

    # initialize empty lists for calculating time for aggregated messages
    init_vars['list_times'] = []
    init_vars['list_aggregates'] = [[],[],[],[],[]]
    init_vars['time_interval_counter'] = 1

    return init_vars

def connect_video_source()->tuple:
    """
    getting video capture for each frame
    """
    print('LOG: connecting to the source')

    cap = cv2.VideoCapture(os.environ['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    return cap, width, height, fps

def load_model():
    """
    loads model depending on library and model size to use
    """
    print('LOG: loading model')

    if os.environ['LIBRARY'] == "ultralytics":
        model_size = load_model_size(os.environ['MODEL_SIZE'], os.environ['LIBRARY'])
        model_ultralytics = YOLO(model_size) # tracking and object detection
        return model_ultralytics

    if os.environ['LIBRARY'] == "yolox":
        if os.environ['MODEL_SIZE'] in ["nano", "tiny"]:
            model_yolox = get_exp(exp_file=None, exp_name=f"yolox-{os.environ['MODEL_SIZE']}")
            return model_yolox
        else:
            model_yolox = get_exp(exp_file=None, exp_name=f"yolox-{os.environ['MODEL_SIZE'][0]}")
            return model_yolox
    return None

def compute_detection(model, frame, labels_dict)->tuple:
    """
    depending on the library used, inference is done on each frame and a list of detected objects of class DetectedObject
    is created and returned together with the inference time

    params
        model: model file to use, obtained with the func load_model(s)
        frame: current frame given by opencv
        labels_dict: dictionary with corresponding labels for categories of objects

    """
    print('LOG: frame read succesfully, computing detection')

    if os.environ['LIBRARY'] == "yolox":
        model_size = load_model_size(os.environ['MODEL_SIZE'],
                                     os.environ['LIBRARY'])
        results_image, img_info = process_frame(model_name=model_size,
                                                exp=model,
                                                frame=frame,
                                                conf=os.environ['CONFIDENCE'],
                                                iou=os.environ['IOU'],
                                                input_size=os.environ['INPUT_SIZE'])
        list_objects = generate_objects(DetectedObject, results_image,
                                        labels_dict, os.environ['LIBRARY'])
        infer_time = img_info['infer_time']

    # calling detection from super-gradients
    if os.environ['LIBRARY'] == "supergradients":
        model_size = load_model_size(os.environ['MODEL_SIZE'],
                                     os.environ['LIBRARY'])
        results_image, infer_time = detect_sg(model_size, frame,
                                              os.environ['CONFIDENCE'], os.environ['IOU'])
        list_objects = generate_objects(DetectedObject, results_image,
                                        labels_dict, os.environ['LIBRARY'])

    if os.environ['LIBRARY'] == "ultralytics":
        input_size = int(os.environ['INPUT_SIZE'].split(',')[0])
        results_ultralytics = model.track(frame,
                                          save=False,
                                          stream=True,
                                          verbose=False,
                                          conf=float(os.environ['CONFIDENCE']),
                                          persist=True,
                                          tracker="botsort.yaml",
                                          iou=float(os.environ['IOU']),
                                          imgsz=input_size,
                                          classes=[0,1,2,3,5,7])

        # results_ultralytics = model(frame,
        #                             save=False,
        #                             stream=True,
        #                             verbose=False,
        #                             conf=float(os.environ['CONFIDENCE']),
        #                             iou=float(os.environ['IOU']),
        #                             imgsz=input_size,
        #                             classes=[0,1,2,3,5,7])

        results_image = []
        infer_time_dict = []
        for r in results_ultralytics:
            results_image.append(r.cpu())
            infer_time_dict.append(r.speed)

        infer_time = (infer_time_dict[0]['preprocess'] +
                      infer_time_dict[0]['inference'] +
                      infer_time_dict[0]['postprocess'])
        infer_time /= 1000

        list_objects = generate_objects(DetectedObject, results_image,
                                        labels_dict, os.environ['LIBRARY'])

    return list_objects, infer_time

def compute_postprocessing(list_objects, frame, init_vars, fps)->tuple:
    """
    performs postprocessing phase using list of detected objects obtained with the inference done
    by the object detection model

    params
        list_objects: list of instatiated objects, corresponds to a detected object
        frame: current frame given by opencv
        init_vars: initial values such as empty dicts and lists for time in zone
        fps: fps of source video
    """
    print('LOG: detection completed, postprocessing frame')
    # show bounding boxes
    if os.environ['DO_DRAW_BBOX'] == ENV_VAR_TRUE_LABEL:
        [obj.draw_boxes(frame, init_vars['labels_dict']) for obj in list_objects]

    # detect man down
    if os.environ['DO_MAN_DOWN'] == ENV_VAR_TRUE_LABEL:
        [obj.get_is_down(frame, os.environ['SHOW_MAN_DOWN']) for obj in list_objects]

    # draw tracks
    if os.environ['DO_DRAW_TRACKS'] == ENV_VAR_TRUE_LABEL:
        [obj.draw_tracks(frame, init_vars['track_history_dict']) for obj in list_objects]

    # for every person inside an area, count the number of frames
    if os.environ['DO_TIME_ZONE'] == ENV_VAR_TRUE_LABEL:
        for obj in list_objects:
            obj_id, obj_is_in_zone, = obj.get_is_in_zone(init_vars['zone_poly'])
            if obj_is_in_zone:
                init_vars['time_in_zone_dict'][obj_id] += 1
        # show time inside zone on top of people's boxes
        [obj.draw_time_zone(frame, init_vars['time_in_zone_dict'], fps, init_vars['zone_poly'],
                            os.environ['SHOW_TIME_ZONE']) for obj in list_objects]

    # count objects
    if os.environ['DO_COUNT_OBJECTS'] == ENV_VAR_TRUE_LABEL:
        number_objs = count_objs(frame, list_objects, os.environ['SHOW_COUNT_PEOPLE'])

    # count people in zone
    if os.environ['DO_COUNT_ZONE'] == ENV_VAR_TRUE_LABEL:
        number_people_zone = count_zone(frame, list_objects, init_vars['zone_poly'],
                                        os.environ['DOOR_COORDS'], os.environ['SHOW_ZONE'])

    # NEEDS TRACKING TODO
    # count people entering and leaving a certain area
    # if os.environ['DO_ENTER_LEAVE'] == ENV_VAR_TRUE_LABEL:
    #     [obj.enter_leave(-----) for obj in list_objects]

    # get a list with the individual info of each detected object
    obj_info = []
    for obj in list_objects:
        x = obj.obj_info()
        obj_info.append(x)

    return number_objs, number_people_zone, obj_info

def detect():
    """
    overall detection function, covers the whole process of object detection of a video source.
    it begins by setting initial values needed for further computations, then it captures the frame of
    a video using opencv, to this frame is applied inference by a yolo model and with the list of
    detected objects, postprocessing techniques with opencv are used to get a better sense of the
    current frame.
    Finally, the information gathered through postprocessing is outputted as a json object, either frame
    by frame or in aggregated form each time a certain interval is reached
    """
    # initial variables needed for different purposes depending on state of the loop
    init_vars = set_initial_vars()
    # video capture and its information
    cap, width, height, fps = connect_video_source()

    # variable to save video output into
    if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
        output = cv2.VideoWriter('output-demo.avi', cv2.VideoWriter_fourcc(*'MPEG'),
                                 fps=fps, frameSize=(width, height))

    # alerting class instance
    # alert = Alert()

    # loading model depending of library to use
    model = load_model()

    # time at the beginning of time interval for aggregation of messages
    time_interval_start = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # read the video source
    while cap.isOpened():
        print('LOG: video source loaded')

        # time at the start of the frame's computation
        start = time.time()

        success, frame = cap.read()
        if success: # frame is read successfully
            # call inference of object detection models
            list_objects, infer_time = compute_detection(model, frame, init_vars['labels_dict'])
            # apply postprocessing to list of detected objects
            number_objs, number_people_zone, obj_info = compute_postprocessing(list_objects,
                                                                               frame,
                                                                               init_vars,
                                                                               fps)
            # write output video
            if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
                output.write(frame)

            # calculating times at end of the computation
            end = time.time()
            elapsed = (end-start)

            # alerting process
            # number_people_down = sum(1 for obj in list_objects if obj.is_down)
            # is_someone_down = number_people_down > 0
            # alert_result = alert.update_man_down(is_someone_down, elapsed)
            # logging.info(f"mail sent {alert_result}")

            # sends message with info only at certain intervals, aggregating results
            if os.environ['MSG_AGGREGATION'] == ENV_VAR_TRUE_LABEL:
                init_vars['list_times'].append(elapsed)
                total_elapsed = sum(init_vars['list_times'])
                # print(init_vars['list_times'])
                # print(total_elapsed)

                # forms a list containing all previous frames informations,
                # goes back to empty after sending aggregated message
                aggregates = aggregate_info(init_vars['list_aggregates'], number_objs,
                                            number_people_zone, list_objects)
                # print(aggregates)

                # sends message when the necessary time has passed
                time_interval_end = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if total_elapsed > int(os.environ['AGGREGATION_TIME_INTERVAL']):
                    print('LOG: results:', '\n')
                    agg_frame_info_dict = send_agg_frame_info(aggregates, time_interval_start,
                                                            time_interval_end, init_vars['time_interval_counter'])
                    time_interval_start = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    init_vars['list_times'] = []
                    init_vars['list_aggregates'] = [[],[],[],[],[]]
                    total_elapsed = 0
                    init_vars['time_interval_counter'] += 1

                    yield agg_frame_info_dict, frame
                else:
                    yield None, frame
            # sends message with info of each frame, about each individual object
            else:
                print('LOG: results:', '\n')
                frame_info_dict = send_frame_info(number_objs, number_people_zone, cap, obj_info)
                yield frame_info_dict, frame

            # calculate and print fps
            print_fps(frame, width, height, infer_time, elapsed)

            # display the annotated frame
            if os.environ['SHOW_IMAGE'] == ENV_VAR_TRUE_LABEL:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # release the video capture object and close the display window
    cap.release()

    if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
        output.release()

    cv2.destroyAllWindows()

    torch.cuda.empty_cache()
    gc.collect()

async def loop_main():
    global current_frame
    while True:
        for frame_info, frame in detect():
            if frame_info is not None:
                frame_info['interval_end'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                frame_info_str = json.dumps(obj=frame_info, indent=4)
                print(frame_info_str, '\n')

if __name__ == "__main__":
    # logging for debugging
    VERBOSITY_LEVEL = logging.INFO
    formatter = logging.Formatter('%(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(VERBOSITY_LEVEL)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(pathname)s \n'+
                               '%(funcName)s (line:%(lineno)d) - '+
                               '%(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("debug.log", mode='w'),
                                  stream_handler])

    # calling generator that yields a json object with info about each frame and the objects in it
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(loop_main())
