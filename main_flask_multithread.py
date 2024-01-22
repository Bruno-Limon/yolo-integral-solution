import os
import numpy as np
import time
import cv2
import threading
from queue import Queue
from flask import Flask, Response, request
from waitress import serve

app = Flask(__name__)

# Global variables for sharing data between threads
frame = None
lock = threading.Lock()
frame_queue = Queue()
fps_queue = Queue()

# Function to capture video frames
def capture_frames():
    global frame

    # Open the video file
    cap = cv2.VideoCapture(os.environ["VIDEO_SOURCE"])
    # cap = cv2.VideoCapture("C:\\Users\\Pavilion\\Desktop\\experiments\\pexels_videos_2431853 (720p).mp4")

    while True:
        # Capture frame-by-frame
        success, current_frame = cap.read()

        # Break the loop if no more frames are available
        if not success:
            break

        # Acquire lock before updating the global variable
        with lock:
            frame = current_frame

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the camera when the loop is exited
    cap.release()

# Function to process frames
def process_frames():
    global frame

    # store fps for each frame until 100th frame
    list_total_detect_times = []

    while True:
        # Acquire lock before accessing the global variable
        with lock:
            if frame is not None:
                t0 = time.time()

                # Perform some processing (e.g., convert to grayscale)
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Put the processed frame into the queue
                frame_queue.put(processed_frame, None)

                t1 = time.time() - t0
                if t1 > 0:
                    current_fps = round((1 / t1), 4)
                    list_total_detect_times.append(current_fps)
                    print(f"frames processed: {len(list_total_detect_times)}/50")

                # yield avg fps at 100th frame
                if len(list_total_detect_times) == 50:
                    avg_detection_time = round(np.mean(list_total_detect_times), 2)
                    print(avg_detection_time)
                    # fps_queue.put(avg_detection_time)
                    list_total_detect_times = []

                # Display the original and processed frames
                cv2.imshow('Processed Frame', processed_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to generate video stream for Flask app
def generate():
    while True:
        # Get a processed frame from the queue
        processed_frame = frame_queue.get()

        # Encode the processed frame as JPEG
        _, buffer = cv2.imencode('.jpg', processed_frame)

        # Convert the frame to bytes and yield it as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

# Video streaming page
@app.route('/video')
def video():
    # Simple login
    login = request.args.get('login')
    # Hardcoded password to login
    if login == 'simple_access1':
        # If success return the stream
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response('Invalid login',mimetype='text/plain')

# Homepage
@app.route('/')
def index():
    return 'Service is up!'


if __name__ == '__main__':

    # Create and start the threads
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_frames)

    capture_thread.start()
    process_thread.start()

    # app.run(host='0.0.0.0', port=5000, debug=False)
    serve(app, host='0.0.0.0', port=5000)

    # Wait for threads to finish
    capture_thread.join()
    process_thread.join()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
