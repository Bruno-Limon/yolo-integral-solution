import cv2
import os
import config
from utils import *

if config.env_vars_local == True:
    from dotenv import load_dotenv
    load_dotenv()

def first_frame_capture(video_source, file_name):
    """
    takes a video as source and saves its first frame
    """
    cap = cv2.VideoCapture(video_source)
    frame_counter = 0

    while cap.isOpened and frame_counter < 1:
        success, frame = cap.read()
        if success:
            cv2.imwrite(file_name, frame)
            frame_counter += 1

def define_area(pixel_frame)->str:
    """
    function that calls opencv's mouse callback to get appropriate coordinates within
    the chosen frame

    param:
        pixel_frame: global var for frame to use
    """
    global pixel_points_coords

    pixel_points_coords = []
    cv2.imshow('image', pixel_frame)
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pixel_points_coords

def click_event(event, x, y, flags, params):
    global pixel_frame
    global pixel_points_coords
    global pixel_type_area

    if len(pixel_points_coords) == 0 and pixel_type_area == "door":
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_instruction = "Place 2 points to form a door. Left click to place point, right click to start over"
        cv2.putText(pixel_frame, text_instruction, (20,40), font, 1, (0, 255, 0), 2)
        cv2.imshow('image', pixel_frame)

    if len(pixel_points_coords) == 0 and pixel_type_area == "area":
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_instruction = "Place 4 points to form an area. Left click to place point, right click to start over"
        cv2.putText(pixel_frame, text_instruction, (20,40), font, 1, (0, 255, 0), 2)
        cv2.imshow('image', pixel_frame)

    if len(pixel_points_coords) == 2 and pixel_type_area == "door":
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_confirmation = "Left click to confirm, right click to start over"
        cv2.putText(pixel_frame, text_confirmation, (20,80), font, 1, (0, 255, 0), 2)
        cv2.imshow('image', pixel_frame)

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyAllWindows()

            # turn into str for env variable
            pixel_points_coords = '|'.join(map(str, pixel_points_coords))
            pixel_points_coords = ''.join(x for x in pixel_points_coords if  x not in '[: :]')
            return pixel_points_coords

        if event == cv2.EVENT_RBUTTONDOWN:
            pixel_points_coords = []
            pixel_frame = cv2.imread(img_path, 1)
            cv2.imshow('image', pixel_frame)
            cv2.setMouseCallback('image', click_event)

    if len(pixel_points_coords) == 4 and pixel_type_area == "area":
        first_point = pixel_points_coords[0]
        last_point = pixel_points_coords[-1]
        cv2.line(pixel_frame, (last_point[0], last_point[1]), (first_point[0], first_point[1]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_confirmation = "Left click to confirm, right click to start over"
        cv2.putText(pixel_frame, text_confirmation, (20,80), font, 1, (0, 255, 0), 2)
        cv2.imshow('image', pixel_frame)

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyAllWindows()

            # turn into str for env variable
            pixel_points_coords = '|'.join(map(str, pixel_points_coords))
            pixel_points_coords = ''.join(x for x in pixel_points_coords if  x not in '[: :]')
            return pixel_points_coords

        if event == cv2.EVENT_RBUTTONDOWN:
            pixel_points_coords = []
            pixel_frame = cv2.imread(img_path, 1)
            cv2.imshow('image', pixel_frame)
            cv2.setMouseCallback('image', click_event)

    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        pixel_points_coords.append([x,y])
        text = '  (' + str(x) + ', ' + str(y) + ')'
        cv2.circle(pixel_frame, (x,y), 2, (0, 255, 0), 6)
        cv2.putText(pixel_frame, text, (x,y), font, 1, (0, 255, 0), 2)
        if len(pixel_points_coords) > 1:
            last_point = pixel_points_coords[-2]
            cv2.line(pixel_frame, (x,y), (last_point[0], last_point[1]), (0, 255, 0), 2)
        cv2.imshow('image', pixel_frame)

    if event == cv2.EVENT_RBUTTONDOWN:
        pixel_points_coords = []
        pixel_frame = cv2.imread(img_path, 1)
        cv2.imshow('image', pixel_frame)
        cv2.setMouseCallback('image', click_event)


if __name__=="__main__":
    file_name = "frame.jpg"
    video_source = os.environ['VIDEO_SOURCE']
    first_frame_capture(video_source, file_name)

    img_path = file_name
    pixel_frame = cv2.imread(img_path, 1)

    # variable to choose the type of coords, either "door" or "area"
    pixel_type_area = "area"
    zone_points = define_area(pixel_frame)

    os.remove(file_name)
    print(zone_points)
