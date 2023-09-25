from flask import Flask, Response, request
from frame_singleton import current_frame
import cv2

app = Flask(__name__)

class Flask_frame:
    def __init__(self, app, singleton_path="frame_singleton.txt", verbose=False):
        self.singleton_path = singleton_path
        self.verbose = verbose
        self.app = app

    # Function to generate video frames from the stream
    def generate_frames(self):
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

    # # Video streaming page
    # @app.route('/video')
    # def video():
    #     # Simple login
    #     login = request.args.get('login')
    #     # Hardcoded password to login
    #     if login == 'simple_access1':
    #         # If success return the stream
    #         return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #     else:
    #         return Response("Invalid login",mimetype='text/plain')

    # # Homepage
    # @app.route('/')
    # def index():
    #     return "Service is up!"

    def run_server(self):
        self.app.run(host='0.0.0.0', port=8080, debug=False)


    # async def loop_main(self):
    #     global current_frame
    #     while True:
    #         for frame_info, frame in detect(vid_path=VIDEO_SOURCE, show_image=SHOW_IMAGE, save_video=SAVE_VIDEO):
    #             print(frame_info, "\n")

    #             # with frame_lock:
    #             current_frame = frame



# flask = Flask_frame(app=app)
# flask.write_file("Hola cico")
# r = flask.read_file()
# print(f":::{r}")
# flask.generate_frames()
