from collections import defaultdict
import os
import gc
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import cv2
import time
import json

from utils import *
load_env_var()

from Detected_object import DetectedObject

# yolo_x through super-gradients
if os.environ['LIBRARY'] == "supergradients":
    from super_gradients_detection import detect_sg

# yolo_x from source
if os.environ['LIBRARY'] == "yolox":
    from YOLOX.yolox.exp import get_exp
    from YOLOX.tools.detect import process_frame

# yolov8 through ultralytics
if os.environ['LIBRARY'] == "ultralytics":
    from ultralytics import YOLO

# enable rtsp capture for opencv
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENV_VAR_TRUE_LABEL = "true"

def detect():
    ####### setting up necessary variables #######
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # dictionary to map the class number obtained with yolo with its name and color for bounding boxes
    labels_dict = get_labels_dict()
    # zone to count people in
    zone_poly = get_zone_poly(os.environ['ZONE_COORDS'])

    # NEEDS TRACKING
    # # dictionaries containing id of people entering/leaving and sets to count them
    # people_entering_dict = {}
    # entering = set()
    # people_leaving_dict = {}
    # leaving = set()

    # NEEDS TRACKING
    # store the track history
    # track_history_dict = defaultdict(lambda: [])
    # store the amount of frames spent inside zone
    time_in_zone_dict = defaultdict(int)

    # information about video source
    frame_counter = 0
    cap = cv2.VideoCapture(os.environ['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # variable to save video output into
    if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
        output = cv2.VideoWriter('output-demo.avi', cv2.VideoWriter_fourcc(*'MPEG'),
                                 fps=fps, frameSize=(width, height))

    # initialize frame skipping mechanism
    is_stream = os.environ['IS_STREAM'] == ENV_VAR_TRUE_LABEL #type(vid_path) is int or vid_path.startswith('rtsp')Z
    frames_to_skip = 1

    # initialize empty lists for calculating time for aggregated messages
    list_times = []
    list_aggregates = [[],[],[],[],[]]
    time_interval_counter = 1

    #######
    if os.environ['LIBRARY'] == "ultralytics":
        model_obj = YOLO("yolov8n.pt") # tracking and object detection

    time_interval_start = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # read the video source
    while cap.isOpened():
        print('LOG: video source loaded')

        # time at the start of the frame's computation
        start = time.time()

        # custom frames skipping
        if os.environ['DO_SKIP_FRAMES'] == ENV_VAR_TRUE_LABEL:
            if not is_stream:
                while frames_to_skip > 0:
                    frames_to_skip -= 1
                    success, frame = cap.read()
                    continue
            else:
                success, frame = cap.read()

            # If failed and stream reconnect
            if not success and is_stream:
                cap = cv2.VideoCapture(os.environ['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
                print('Reconnecting to the stream')
                continue

            # If failed and not stream -> video finished
            elif not success:
                break
        else:
            success, frame = cap.read()

        # frame is read successfully
        if success:
            print('LOG: video detected succesfully')

            # in case of using yolox without super-gradients
            # model = get_exp(exp_file=None, exp_name='yolox-nano')
            # results_image, img_info = process_frame(model_name='src/models/yolox_nano.pth', exp=model, frame=frame)

            # calling detection from super-gradients
            if os.environ['LIBRARY'] == "supergradients":
                results_image, infer_time = detect_sg(os.environ['MODEL_ID'], frame)
                list_objects = generate_objects(DetectedObject, results_image, labels_dict, os.environ['LIBRARY'])

            if os.environ['LIBRARY'] == "ultralytics":
                results_ultralytics = model_obj.track(frame, save=False, stream=True, verbose=False, conf=.4,
                                                      persist=True, tracker="botsort.yaml", iou=.5, classes=[0,1,2,3,5,7])

                results_image = []
                infer_time_dict = []
                for r in results_ultralytics:
                    results_image.append(r.cpu())
                    infer_time_dict.append(r.speed)

                infer_time = (infer_time_dict[0]['preprocess'] +
                              infer_time_dict[0]['inference'] +
                              infer_time_dict[0]['postprocess'])
                infer_time /= 100

                list_objects = generate_objects(DetectedObject, results_image, labels_dict, os.environ['LIBRARY'])

            # show bounding boxes
            if os.environ['DO_DRAW_BBOX'] == ENV_VAR_TRUE_LABEL:
                [obj.draw_boxes(frame, labels_dict) for obj in list_objects]

            # detect man down
            if os.environ['DO_MAN_DOWN'] == ENV_VAR_TRUE_LABEL:
                [obj.get_is_down(frame, os.environ['SHOW_MAN_DOWN']) for obj in list_objects]

            # NEEDS TRACKING
            # draw tracks
            # if os.environ['DO_DRAW_TRACKS'] == ENV_VAR_TRUE_LABEL:
            #     [obj.draw_tracks(frame, track_history_dict) for obj in list_objects]

            # NEEDS TRACKING
            # for every person inside an area, count the number of frames
            if os.environ['DO_TIME_ZONE'] == ENV_VAR_TRUE_LABEL:
                for obj in list_objects:
                    obj_id, obj_is_in_zone, = obj.get_is_in_zone(zone_poly)
                    if obj_is_in_zone:
                        time_in_zone_dict[obj_id] += 1
                # show time inside zone on top of people's boxes
                [obj.draw_time_zone(frame, time_in_zone_dict, fps, zone_poly, os.environ['SHOW_TIME_ZONE']) for obj in list_objects]

            # count objects
            if os.environ['DO_COUNT_OBJECTS'] == ENV_VAR_TRUE_LABEL:
                number_objs = count_objs(frame, list_objects, os.environ['SHOW_COUNT_PEOPLE'])

            # count people in zone
            if os.environ['DO_COUNT_ZONE'] == ENV_VAR_TRUE_LABEL:
                number_people_zone = count_zone(frame, list_objects, zone_poly, os.environ['DOOR_COORDS'], os.environ['SHOW_ZONE'])

            # NEEDS TRACKING
            # # count people entering and leaving a certain area
            # if do_enter_leave == True:
                # [obj.enter_leave(frame, width, show_enter_leave) for obj in list_objects]

            # get a list with the individual info of each detected object
            obj_info = []
            for obj in list_objects:
                x = obj.obj_info()
                obj_info.append(x)

            # write output video
            if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
                output.write(frame)

            # display the annotated frame
            if os.environ['SHOW_IMAGE'] == ENV_VAR_TRUE_LABEL:
                cv2.imshow('Demo', frame)

        # calculating times at end of the computation
        end = time.time()
        elapsed = (end-start)

        # sends message with info of each frame, about each individual object
        # if os.environ['DO_MSG_BY_FRAME'] == ENV_VAR_TRUE_LABEL:
        #     print('LOG: results:', '\n')
        #     frame_info_dict = send_frame_info(number_objs, number_people_zone, cap, obj_info)
        #     yield frame_info_dict, frame

        # sends message with info only at certain intervals, aggregating results
        if os.environ['MSG_AGGREGATION'] == ENV_VAR_TRUE_LABEL:
            list_times.append(elapsed)
            total_elapsed = sum(list_times)
            # print(list_times)
            # print(total_elapsed)

            # forms a list containing all previous frames informations, goes back to empty after sending aggregated message
            aggregates = aggregate_info(list_aggregates, number_objs, number_people_zone, list_objects)
            # print(aggregates)

            # sends message when the necessary time has passed
            time_interval_end = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if total_elapsed > int(os.environ['TIME_INTERVAL']):
                print('LOG: results:', '\n')
                agg_frame_info_dict = send_agg_frame_info(aggregates, time_interval_start, time_interval_end, time_interval_counter)
                time_interval_start = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                list_times = []
                list_aggregates = [[],[],[],[],[]]
                total_elapsed = 0
                time_interval_counter += 1

                yield agg_frame_info_dict, frame
            else:
                yield None, frame
        # sends message with info of each frame, about each individual object
        else:
            print('LOG: results:', '\n')
            frame_info_dict = send_frame_info(number_objs, number_people_zone, cap, obj_info)
            yield frame_info_dict, frame


        print_fps(frame, width, height, infer_time, elapsed)

        # calculating times at end of the computation
        end = time.time()
        elapsed = (end-start)
        frames_to_skip=int(fps*elapsed)

    # release the video capture object and close the display window
    cap.release()

    if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
        output.release()

    cv2.destroyAllWindows()

    torch.cuda.empty_cache()
    gc.collect()


############## FLASK #################

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
        print('Client disconnected')

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
        return Response('Invalid login',mimetype='text/plain')

# Homepage
@app.route('/')
def index():
    return 'Service is up!'

def run_server():
    app.run(host='0.0.0.0', port=os.environ['FLASK_PORT'], debug=False)

async def loop_main():
    global current_frame
    while True:
        for frame_info, frame in detect():
            if frame_info is not None:
                frame_info['interval_end'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                frame_info['device_id'] = os.environ['DEVICE_ID']
                frame_info['camera_id'] = os.environ['CAMERA_ID']
                frame_info['model_id'] = os.environ['MODEL_ID']
                frame_info_str = json.dumps(obj=frame_info, indent=4)
                print(frame_info_str, '\n')

            # with frame_lock:
            current_frame = frame

############## FLASK #################


if __name__ == "__main__":
    from frame_singleton import current_frame

    if os.environ['EXPOSE_STREAM'] == ENV_VAR_TRUE_LABEL:
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

    # calling generator that yields a json object with info about each frame and the objects in it
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(loop_main())