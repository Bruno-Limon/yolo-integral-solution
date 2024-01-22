import os
import gc
import time
from datetime import datetime
import numpy as np
import json
import logging
import threading
import torch
import cv2
import asyncio
from flask import Flask, Response, request, g
# from waitress import serve

from classes.alert import Alert
from utils.frame_singleton import current_frame

from utils.detection_utils import set_initial_vars, load_model, compute_detection
from utils.detection_utils import compute_postprocessing, connect_video_source, send_alert
from utils.postprocessing_utils import send_alert_info, send_agg_frame_info
from utils.postprocessing_utils import send_frame_info, aggregate_info, print_fps


from config import enable_local_work, config_tracker
enable_local_work()
config_tracker()

# enable rtsp capture for opencv
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENV_VAR_TRUE_LABEL = "true"

def detect():
    """
    Overall detection function, covers the whole process of object detection of a video source.
    it begins by setting initial values needed for further computations, then it captures the frame
    of a video using opencv, to this frame is applied inference by a yolo model and with the list
    of detected objects, postprocessing techniques with opencv are used to get a better sense of
    the current frame.
    Finally, the information gathered through postprocessing is outputted as a json object,
    either frame by frame or in aggregated form each time a certain interval is reached
    """
    # list to store
    list_total_detect_times = []

    # initial variables needed for different purposes depending on state of the loop
    print('LOG: clearing cuda', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    init_vars = set_initial_vars()

    # video capture and its information
    print('LOG: connecting to the source', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    cap, width, height, fps, read_frame_time = connect_video_source()

    # variable to save video output into
    if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
        output = cv2.VideoWriter('src/data/output-demo.avi', cv2.VideoWriter_fourcc(*'MPEG'),
                                 fps=fps, frameSize=(width, height))

    # alerting class instance
    if os.environ['ALERT_SEND'] == ENV_VAR_TRUE_LABEL:
        alerts_dict = {}
        alerts_dict['man_down'] = Alert(alert_type='man down')
        alerts_dict['too_many_people'] = Alert(alert_type='people in zone')

    # loading model depending of library to use
    print('LOG: loading model', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    model_detection, model_pose = load_model()

    # time at the beginning of time interval for aggregation of messages
    time_interval_start = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # read the video source
    while cap.isOpened():
        print('LOG: video source loaded', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

        # time at the start of the frame's computation
        frame_time_start = time.time()

        """
        Skipping frames
        """
        if os.environ['DO_SKIP_FRAMES'] == ENV_VAR_TRUE_LABEL:
            print('LOG: skipping missed frames', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            if not init_vars['is_stream']:
                while init_vars['frames_to_skip'] > 0:
                    init_vars['frames_to_skip'] -= 1
                    start_time_read = time.time()
                    success, frame = cap.read()
                    end_time_read = (time.time() - start_time_read)
                    continue
            else:
                start_time_read = time.time()
                success, frame = cap.read()
                end_time_read = (time.time() - start_time_read)

            # If failed and stream reconnect
            if not success and init_vars['is_stream']:
                cap = cv2.VideoCapture(os.environ['VIDEO_SOURCE'], cv2.CAP_FFMPEG)
                print('Reconnecting to the stream')
                continue

            # If failed and not stream -> video finished
            elif not success:
                break
        else:
            start_time_read = time.time()
            success, frame = cap.read()
            end_time_read = (time.time() - start_time_read)
        init_vars['times_dict']['read_frame_time'] = end_time_read


        """
        Frame is read succesfully, so detection is done with the chosen model and then
        the results are processed to get desired information"""
        if success:
            # call inference of object detection models
            print('LOG: frame read succesfully, computing detection', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            list_objects, results_pose, infer_time = compute_detection(model_detection,
                                                                       model_pose,
                                                                       frame,
                                                                       init_vars['labels_dict'])
            init_vars['times_dict']['infer_time'] = infer_time

            print('LOG: detection completed, postprocessing frame', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            # apply postprocessing to list of detected objects
            results_postproc, post_process_time, frame  = compute_postprocessing(list_objects,
                                                                                 results_pose,
                                                                                 frame,
                                                                                 init_vars,
                                                                                 cap)
            init_vars['times_dict']['post_process_time'] = post_process_time

            # write output video
            if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
                output.write(frame)

            # calculating times at end of the computation
            detection_time_end = time.time()
            detection_time_elapsed = (detection_time_end-frame_time_start)
            init_vars['times_dict']['total_time'] = detection_time_elapsed
            list_total_detect_times.append(detection_time_elapsed)
            print(f"frames processed: {len(list_total_detect_times)}/100")

            if len(list_total_detect_times) == 100:
                avg_detection_time = round(np.mean(list_total_detect_times),4)
                list_total_detect_times = []
                yield None, None, frame, avg_detection_time

            """
            Alerting
            """
            alert_result = None
            if os.environ['ALERT_SEND'] == ENV_VAR_TRUE_LABEL:
                alert_result, alert_type, alert_id = send_alert(alerts_dict,
                                                                list_objects,
                                                                results_postproc,
                                                                detection_time_elapsed)
            """
            Sends message with info only at certain intervals, aggregating results for each frame
            and then sending them at certain time intervals.
            Else, it sends info about each frame.
            In case of alert, it also sends a message with it.
            """
            agg_frame_info_dict = None
            alert_info_dict = None

            if alert_result:
                alert_info_dict = send_alert_info(alert_type, alert_id)
                yield agg_frame_info_dict, alert_info_dict, frame, None

            if os.environ['MSG_AGGREGATION'] == ENV_VAR_TRUE_LABEL:
                init_vars['list_times'].append(detection_time_elapsed)
                time_elapsed = sum(init_vars['list_times'])

                # aggregates frame info over time
                aggregates = aggregate_info(init_vars['list_aggregates'],
                                            results_postproc,
                                            list_objects)

                time_interval_end = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if time_elapsed > int(os.environ['AGGREGATION_TIME_INTERVAL']):
                    print('LOG: results:', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], '\n')
                    agg_frame_info_dict = send_agg_frame_info(aggregates,
                                                              time_interval_start,
                                                              time_interval_end,
                                                              init_vars['time_interval_counter'])
                    time_interval_start = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    init_vars['list_times'] = []
                    init_vars['list_aggregates'] = [[],[],[],[],[],
                                                    [],[],[],[]]
                    time_elapsed = 0
                    init_vars['time_interval_counter'] += 1

                    yield agg_frame_info_dict, None, frame, None
                else:
                    yield agg_frame_info_dict, None, frame, None
            # sends message with info of each frame, about each individual object
            else:
                print('LOG: results:', datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], '\n')
                frame_info_dict = send_frame_info(results_postproc, cap)
                yield frame_info_dict, None, frame, None

            """
            End of detection phase, object recognition and its succesive postprocessing
            is done, yielding a dictionary with the information about the current frame.
            Now passing to next frame
            """
            # calculate and print fps
            print_fps(frame, width, height, init_vars['times_dict'])

            # display the frame with visual information
            if os.environ['SHOW_IMAGE'] == ENV_VAR_TRUE_LABEL:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # calculating times at end of frame
            frame_time_end = time.time()
            total_time_elapsed = (frame_time_end-frame_time_start)
            init_vars['frames_to_skip'] = int(fps*total_time_elapsed)

        else:
            break

    # release the video capture object and close the display window
    cap.release()

    if os.environ['SAVE_VIDEO'] == ENV_VAR_TRUE_LABEL:
        output.release()

    cv2.destroyAllWindows()

    torch.cuda.empty_cache()
    gc.collect()


"""
FLASK application
"""
app = Flask(__name__)

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
    # serve(app, host='0.0.0.0', port=os.environ['FLASK_PORT'])
    app.run(host='0.0.0.0', port=os.environ['FLASK_PORT'], debug=False)


async def loop_main():
    global current_frame
    while True:
        for frame_info, alert_info, frame, avg_detection_time in detect():
            if frame_info:
                frame_info['device_id'] = os.environ['DEVICE_ID']
                frame_info['camera_id'] = os.environ['CAMERA_ID']
                frame_info['model_id'] = os.environ['MODEL_ID']
                frame_info_str = json.dumps(obj=frame_info, indent=4)
                print(frame_info_str, '\n')

            if alert_info:
                alert_info['device_id'] = os.environ['DEVICE_ID']
                alert_info_str = json.dumps(obj=alert_info, indent=4)
                print(alert_info_str, '\n')

            if avg_detection_time:
                print(f"Avg detection time: {avg_detection_time}")

            # with frame_lock:
            current_frame = frame

def main_noloop():
    for frame_info, alert_info, frame, avg_detection_time in detect():
            if frame_info:
                frame_info['device_id'] = os.environ['DEVICE_ID']
                frame_info['camera_id'] = os.environ['CAMERA_ID']
                frame_info['model_id'] = os.environ['MODEL_ID']
                frame_info_str = json.dumps(obj=frame_info, indent=4)
                print(frame_info_str, '\n')

            if alert_info:
                alert_info['device_id'] = os.environ['DEVICE_ID']
                alert_info_str = json.dumps(obj=alert_info, indent=4)
                print(alert_info_str, '\n')

            if avg_detection_time:
                print(f"Avg detection time: {avg_detection_time}")

            # with frame_lock:
            current_frame = frame


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

    if os.environ['EXPOSE_STREAM'] == ENV_VAR_TRUE_LABEL:
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

    # calling generator that yields a json object with info about each frame and the objects in it
    if os.environ['VIDEO_LOOP'] == ENV_VAR_TRUE_LABEL:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(loop_main())
    else:
        main_noloop()