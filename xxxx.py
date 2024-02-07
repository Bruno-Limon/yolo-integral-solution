from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("C:\\Users\\Pavilion\\Desktop\\Data\\vid15.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
write_video = True
if write_video:
    video_writer = cv2.VideoWriter("heatmap_output.avi",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))

# Init heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=w,
                     imh=h,
                     view_img=True,
                     shape="circle",
                     decay_factor=.9999999,
                     heatmap_alpha=.5)

while cap.isOpened():
    success, im0 = cap.read()
    tracks = model.track(im0,
                         conf=.01,
                         persist=True,
                         tracker="src/tracker.yaml",
                         iou=.45,
                         imgsz=640,
                         agnostic_nms=True,
                         vid_stride=False,
                         )
    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    if write_video:
        video_writer.write(im0)

cap.release()
if write_video:
    video_writer.release()
cv2.destroyAllWindows()