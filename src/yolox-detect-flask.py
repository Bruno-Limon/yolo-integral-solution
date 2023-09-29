from collections import defaultdict
import os
import gc

import torch
import cv2
import time
import json

from utils import *
from load_env_var import load_var
from super_gradients_detection import detect_sg
from Detected_object import DetectedObject

# from YOLOX.yolox.exp import get_exp
# from YOLOX.tools.detect import process_frame

def detect(environment_variables):
    # enable rtsp capture for opencv
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # dictionary to map the class number obtained with yolo with its name and color for bounding boxes
    labels_dict = get_labels_dict()
    # zone to count people in
    zone_poly = get_zone_poly(environment_variables['ZONE_COORDS'])
    door_poly = get_zone_poly(environment_variables['DOOR_COORDS'])

    LOG_KW = 'LOG'
    print(f'{LOG_KW}: model loaded, starting detection')

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

    frame_counter = 0
    cap = cv2.VideoCapture(environment_variables['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if environment_variables['SAVE_VIDEO'] == 'True':
        output = cv2.VideoWriter('output-demo.avi', cv2.VideoWriter_fourcc(*'MPEG'),
                                 fps=fps, frameSize=(width, height))

    # Initialize frame skipping mechanism
    is_stream = False #type(vid_path) is int or vid_path.startswith('rtsp')Z
    frames_to_skip = 1
    # Read the whole input
    while cap.isOpened():
        print('Connection established')
        # Custom frames skipping
        start = time.time()
        do_skip_frames = environment_variables['DO_SKIP_FRAMES']
        if do_skip_frames == 'True':
            if not is_stream:
                while frames_to_skip > 0:
                    frames_to_skip -= 1
                    success, frame = cap.read()
                    continue
            else:
                success, frame = cap.read()

            # If failed and stream reconnect
            if not success and is_stream:
                cap = cv2.VideoCapture(environment_variables['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
                print('Reconnecting to the stream')
                continue
            # If failed and not stream -> video finished
            elif not success:
                break
        else:
            success, frame = cap.read()
        # If frame is read, compute outputs
        if success:
            print(f'{LOG_KW}: video detected')
            # model = get_exp(exp_file=None, exp_name='yolox-nano')
            # results_image, img_info = process_frame(model_name='src/models/yolox_nano.pth', exp=model, frame=frame)
            results_image, infer_time = detect_sg(frame)
            list_objects = generate_objects(DetectedObject, results_image, labels_dict)

            # show bounding boxes
            if environment_variables['DO_DRAW_BBOX'] == 'True':
                [obj.draw_boxes(frame, labels_dict) for obj in list_objects]

            # detect man down
            if environment_variables['DO_MAN_DOWN'] == 'True':
                [obj.get_is_down(frame, environment_variables['SHOW_MAN_DOWN']) for obj in list_objects]

            # NEEDS TRACKING
            # draw tracks
            # if environment_variables['DO_DRAW_TRACKS'] == 'True':
            #     [obj.draw_tracks(frame, track_history_dict) for obj in list_objects]

            # NEEDS TRACKING
            # for every person inside an area, count the number of frames
            if environment_variables['DO_TIME_ZONE'] == 'True':
                for obj in list_objects:
                    obj_id, obj_is_in_zone, = obj.get_is_in_zone(zone_poly)
                    if obj_is_in_zone:
                        time_in_zone_dict[obj_id] += 1
                # show time inside zone on top of people's boxes
                [obj.draw_time_zone(frame, time_in_zone_dict, fps, zone_poly, environment_variables['SHOW_TIME_ZONE']) for obj in list_objects]

            # count objects
            if environment_variables['DO_COUNT_OBJECTS'] == 'True':
                number_objs = count_objs(frame, list_objects, environment_variables['SHOW_COUNT_PEOPLE'])

            # count people in zone
            if environment_variables['DO_COUNT_ZONE'] == 'True':
                number_people_zone = count_zone(frame, list_objects, zone_poly, door_poly, environment_variables['SHOW_ZONE'])

            # NEEDS TRACKING
            # # count people entering and leaving a certain area
            # if do_enter_leave == True:
                # [obj.enter_leave(frame, width, show_enter_leave) for obj in list_objects]

            # get object info
            obj_info = []
            for obj in list_objects:
                x = obj.obj_info()
                obj_info.append(x)

            # get frame info
            print(f'{LOG_KW}: results:', '\n')
            frame_info_dict = send_frame_info(number_objs, number_people_zone, cap, obj_info)
            yield frame_info_dict, frame

            # write output video
            if environment_variables['SAVE_VIDEO'] == 'True':
                output.write(frame)

            # display the annotated frame
            if environment_variables['SHOW_IMAGE'] == 'True':
                cv2.imshow('Demo', frame)

            # break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        end = time.time()
        elapsed = (end-start)
        frames_to_skip=int(fps*elapsed)

        print_fps(frame, width, height, infer_time, elapsed)

    # release the video capture object and close the display window
    cap.release()

    if environment_variables['SAVE_VIDEO'] == 'True':
        output.release()

    cv2.destroyAllWindows()

    # del model_obj
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
    app.run(host='0.0.0.0', port=environment_variables['FLASK_PORT'], debug=False)

async def loop_main():
    global current_frame
    while True:
        for frame_info, frame in detect(environment_variables):

            frame_info['device_id'] = environment_variables['DEVICE_ID']
            frame_info['camera_id'] = environment_variables['CAMERA_ID']
            frame_info['model_id'] = environment_variables['MODEL_ID']
            frame_info_str = json.dumps(obj=frame_info, indent=4)
            print(frame_info_str, '\n')

            # with frame_lock:
            current_frame = frame

############## FLASK #################


if __name__ == "__main__":

    from frame_singleton import current_frame
    environment_variables = load_var(iothub=False)

    if environment_variables['EXPOSE_STREAM'] == 'True':
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

    # calling generator that yields a json object with info about each frame and the objects in it
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(loop_main())

    #'rtsp://admin:T0lstenc088@abyss88.ignorelist.com/1'