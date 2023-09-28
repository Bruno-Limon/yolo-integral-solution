from collections import defaultdict
import os
import gc

import torch
import cv2
import time

from utils import *
from super_gradients_detection import detect_sg
from Detected_object import DetectedObject

# from YOLOX.yolox.exp import get_exp
# from YOLOX.tools.detect import process_frame

# enable rtsp capture for opencv
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def detect(vid_path, show_image, save_video, zone_coords, # params
           do_draw_bbox, do_man_down, do_draw_tracks, do_time_zone, do_count_objects, do_count_zone, do_enter_leave, # tasks to do
           show_man_down, show_zone, show_count_people, show_time_zone, show_enter_leave): # show on frame
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    is_stream = True #type(vid_path) is int or vid_path.startswith('rtsp')

    # dictionary to map the class number obtained with yolo with its name and color for bounding boxes
    labels_dict = get_labels_dict()
    # zone to count people in
    zone_poly = get_zone_poly(zone_coords)

    LOG_KW = "LOG"
    print(f"{LOG_KW}: model loaded, starting detection")

    # NEEDS TRACKING
    # # dictionaries containing id of people entering/leaving and sets to count them
    # people_entering_dict = {}
    # entering = set()
    # people_leaving_dict = {}
    # leaving = set()

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
            # model = get_exp(exp_file=None, exp_name="yolox-nano")
            # results_image, img_info = process_frame(model_name="src/models/yolox_nano.pth", exp=model, frame=frame)
            results_image, infer_time = detect_sg(frame)
            list_objects = generate_objects(DetectedObject, results_image, labels_dict)

            # show bounding boxes
            if do_draw_bbox == True:
                [obj.draw_boxes(frame, labels_dict) for obj in list_objects]

            # detect man down
            if do_man_down == True:
                [obj.get_is_down(frame, show_man_down) for obj in list_objects]

            # NEEDS TRACKING
            # draw tracks
            # if do_draw_tracks == True:
            #     [obj.draw_tracks(frame, track_history_dict) for obj in list_objects]

            # NEEDS TRACKING
            # for every person inside an area, count the number of frames
            if do_time_zone == True:
                for obj in list_objects:
                    obj_id, obj_is_in_zone, = obj.get_is_in_zone(zone_poly)
                    if obj_is_in_zone:
                        time_in_zone_dict[obj_id] += 1
                # show time inside zone on top of people's boxes
                [obj.draw_time_zone(frame, time_in_zone_dict, fps, zone_poly, show_time_zone) for obj in list_objects]

            # count objects
            if do_count_objects == True:
                number_objs = count_objs(frame, list_objects, show_count_people)

            # count people in zone
            if do_count_zone == True:
                number_people_zone = count_zone(frame, list_objects, zone_poly, show_zone)

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

        print_fps(frame, width, height, infer_time, elapsed)

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
        for frame_info, frame in detect(vid_path=VIDEO_SOURCE,
                                        show_image=SHOW_IMAGE,
                                        save_video=SAVE_VIDEO,
                                        zone_coords=ZONE_COORDS,
                                        # tasks
                                        do_draw_bbox=DO_DRAW_BBOX,
                                        do_man_down=DO_MAN_DOWN,
                                        do_draw_tracks=DO_DRAW_TRACKS,
                                        do_time_zone=DO_TIME_ZONE,
                                        do_count_objects=DO_COUNT_OBJECTS,
                                        do_count_zone=DO_COUNT_ZONE,
                                        do_enter_leave=DO_ENTER_LEAVE,
                                        # visual
                                        show_man_down=SHOW_MAN_DOWN,
                                        show_zone=SHOW_ZONE,
                                        show_count_people=SHOW_COUNT_PEOPLE,
                                        show_time_zone=SHOW_TIME_ZONE,
                                        show_enter_leave=SHOW_ENTER_LEAVE):
            print(frame_info, "\n")

            # with frame_lock:
            current_frame = frame

############## FLASK #################


if __name__ == "__main__":

    # load environment variables
    # VIDEO_SOURCE = 'http://185.137.146.14:80/mjpg/video.mjpg' #os.getenv(key='VIDEO_SOURCE')
    VIDEO_SOURCE = 'https://nvidia.box.com/shared/static/veuuimq6pwvd62p9fresqhrrmfqz0e2f.mp4' #os.getenv(key='VIDEO_SOURCE')
    SHOW_IMAGE = False #os.getenv(key='SHOW_IMAGE')
    SAVE_VIDEO = False #os.getenv(key='SAVE_VIDEO')
    EXPOSE_STREAM = True #os.getenv(key='EXPOSE_STREAM')
    RUN_WAIT_TIME = 100 #int(os.getenv(key='RUN_WAIT_TIME'))
    FLASK_PORT = 8080 #os.getenv(key='FLASK_PORT')

    # tasks that can be done
    DO_DRAW_BBOX = True #os.getenv(key='DO_DRAW_BBOX')
    DO_MAN_DOWN = False #os.getenv(key='DO_MAN_DOWN')
    DO_DRAW_TRACKS = False #os.getenv(key='DO_DRAW_TRACKS')
    DO_TIME_ZONE = False #os.getenv(key='DO_TIME_ZONE')
    DO_COUNT_OBJECTS = True #os.getenv(key='DO_COUNT_OBJECTS')
    DO_COUNT_ZONE = True #os.getenv(key='DO_COUNT_ZONE')
    DO_ENTER_LEAVE = False #os.getenv(key='DO_ENTER_LEAVE')

    # visual elements that can be shown on frame
    SHOW_MAN_DOWN = False #os.getenv(key='SHOW_MAN_DOWN')
    SHOW_ZONE = True #os.getenv(key='SHOW_ZONE')
    SHOW_COUNT_PEOPLE = True #os.getenv(key='SHOW_COUNT_PEOPLE')
    SHOW_TIME_ZONE = False #os.getenv(key='SHOW_TIME_ZONE')
    SHOW_ENTER_LEAVE = False #os.getenv(key='SHOW_ENTER_LEAVING')

    ZONE_COORDS = "493,160|780,200|580,380|200,280" #os.getenv(key='ZONE_COORDS')
    DOOR_COORDS = "200,280|440,355|430,365|185,290" #os.getenv(key='DOOR_COORDS')
    DOOR2_COORDS = "180,300|420,375|410,385|165,310" #os.getenv(key='DOOR2_COORDS')

    if EXPOSE_STREAM:
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

    # calling generator that yields a json object with info about each frame and the objects in it
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(loop_main())

    #'rtsp://admin:T0lstenc088@abyss88.ignorelist.com/1'