import cv2

def define_zone(get_pixel_frame):
    global get_pixel_points
    get_pixel_points = []
    cv2.imshow('image', get_pixel_frame)
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return get_pixel_points


def click_event(event, x, y, flags, params):
    global get_pixel_frame
    global get_pixel_points

    if len(get_pixel_points) == 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_instruction = "Place 4 points to form an area. Left click to place point, right click to start over"
        cv2.putText(get_pixel_frame, text_instruction, (20,40), font, 1, (0, 255, 0), 2)
        cv2.imshow('image', get_pixel_frame)

    if len(get_pixel_points) == 4:
        first_point = get_pixel_points[0]
        last_point = get_pixel_points[-1]
        cv2.line(get_pixel_frame, (last_point[0], last_point[1]), (first_point[0], first_point[1]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_confirmation = "Left click to confirm, right click to start over"
        cv2.putText(get_pixel_frame, text_confirmation, (20,80), font, 1, (0, 255, 0), 2)
        cv2.imshow('image', get_pixel_frame)

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyAllWindows()
            return get_pixel_points

        if event == cv2.EVENT_RBUTTONDOWN:
            get_pixel_points = []
            get_pixel_frame = cv2.imread(img_path, 1)
            cv2.imshow('image', get_pixel_frame)
            cv2.setMouseCallback('image', click_event)

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)

        font = cv2.FONT_HERSHEY_SIMPLEX
        get_pixel_points.append([x,y])
        text = '  (' + str(x) + ', ' + str(y) + ')'
        cv2.circle(get_pixel_frame, (x,y), 2, (0, 255, 0), 6)
        cv2.putText(get_pixel_frame, text, (x,y), font, 1, (0, 255, 0), 2)
        if len(get_pixel_points) > 1:
            last_point = get_pixel_points[-2]
            cv2.line(get_pixel_frame, (x,y), (last_point[0], last_point[1]), (0, 255, 0), 2)
        cv2.imshow('image', get_pixel_frame)

    if event == cv2.EVENT_RBUTTONDOWN:
        get_pixel_points = []
        get_pixel_frame = cv2.imread(img_path, 1)
        cv2.imshow('image', get_pixel_frame)
        cv2.setMouseCallback('image', click_event)


if __name__=="__main__":

    img_path = 'src/img1.PNG'
    get_pixel_frame = cv2.imread(img_path, 1)
    zone_points = define_zone(get_pixel_frame)
    print(zone_points)
