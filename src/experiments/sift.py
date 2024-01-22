import cv2
import matplotlib.pyplot as plt

# reading image using the imread() function
imageread = cv2.imread('src/experiments/1.jpg')
imagegray = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

features = cv2.SIFT_create()

keypoints = features.detect(imagegray, None)

# drawKeypoints function is used to draw keypoints
output_image = cv2.drawKeypoints(imagegray, keypoints, 0, (0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(output_image)
plt.show()