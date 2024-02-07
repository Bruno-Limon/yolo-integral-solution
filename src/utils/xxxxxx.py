import cv2
import numpy as np

def get_nearest_points(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    points_bbox1 = np.array([(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)])
    points_bbox2 = np.array([(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)])

    distances = np.linalg.norm(points_bbox1[:, np.newaxis, :] - points_bbox2, axis=2)
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)

    nearest_point_bbox1 = tuple(points_bbox1[min_indices[0]])
    nearest_point_bbox2 = tuple(points_bbox2[min_indices[1]])

    return nearest_point_bbox1, nearest_point_bbox2

def draw_nearest_points(frame, bboxes):
    # Draw bounding boxes
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw lines connecting nearest points
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            nearest_point1, nearest_point2 = get_nearest_points(bboxes[i], bboxes[j])
            distance = np.sqrt((nearest_point2[0] - nearest_point1[0])**2 +
                               (nearest_point2[1] - nearest_point1[1])**2)
            print(i,j,distance)
            cv2.line(frame, nearest_point1, nearest_point2, (0, 0, 255), 2)

# Example usage
frame = cv2.imread("C:\\Users\\Pavilion\\Desktop\\data\\x.PNG")
bboxes = [(919, 389, 448, 176), (45, 243, 90, 103), (889, 138, 85, 43), (1705, 285, 46, 86)]

draw_nearest_points(frame, bboxes)

cv2.imshow('Nearest Points', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
