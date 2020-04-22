import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('topic', type=str,
                    help='A required integer topic argument')
args = parser.parse_args()

img = cv2.imread(args.topic)
grey = cv2.imread(args.topic, cv2.IMREAD_GRAYSCALE)
traffic_light = np.array(
    255 * (grey / 255) ** 2.0, dtype="uint8"
)
traffic_light = np.array(traffic_light / np.max(traffic_light) * 255, dtype="uint8")

cv2.namedWindow("img_hsv", cv2.WINDOW_NORMAL)
cv2.imshow("img_hsv", traffic_light)

print(traffic_light.shape)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# RED: lower mask (0-10)
lower_red = np.array([0,0,0])
upper_red = np.array([10,255,255])
mask_red0 = cv2.inRange(img_hsv, lower_red, upper_red)

# RED: upper mask (170-180)
lower_red = np.array([160,0,0])
upper_red = np.array([180,255,255])
mask_red1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join RED masks
mask_red = mask_red0+mask_red1

# YELLOW: mask (12-36)
lower_yellow = np.array([10,0,0])
upper_yellow = np.array([32,255,255])
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# GREEN: mask (45-95)
lower_green = np.array([45,0,0])
upper_green = np.array([95,255,255])
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

area_red = cv2.countNonZero(mask_red)
area_yellow = cv2.countNonZero(mask_yellow)
area_green = cv2.countNonZero(mask_green)

color_map = ["RED", "YELLOW", "GREEN"]
print([area_red, area_yellow, area_green])
detected_color = color_map[np.argmax([area_red, area_yellow, area_green])]
print(detected_color)

if (detected_color == "RED"):
	cv2.namedWindow("mask_red", cv2.WINDOW_NORMAL)
	cv2.imshow("mask_red", mask_red)

if (detected_color == "YELLOW"):
	cv2.namedWindow("mask_yellow", cv2.WINDOW_NORMAL)
	cv2.imshow("mask_yellow", mask_yellow)

if (detected_color == "GREEN"):
	cv2.namedWindow("mask_green", cv2.WINDOW_NORMAL)
	cv2.imshow("mask_green", mask_green)

cv2.waitKey(0)
cv2.destroyAllWindows()
