import cv2
import numpy as np

#jpg, png
#img = cv2.imread('test/red.png')
#img = cv2.imread('test/yellow.png') 
img = cv2.imread('test/green.png')

h, w = img.shape[:2]

img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# RED: lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask_red0 = cv2.inRange(img_hsv, lower_red, upper_red)

# RED: upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask_red1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join RED masks
mask_red = mask_red0+mask_red1

# YELLOW: mask (12-36)
lower_yellow = np.array([12,50,50])
upper_yellow = np.array([36,255,255])
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# GREEN: mask (45-95)
lower_green = np.array([45,50,50])
upper_green = np.array([95,255,255])
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

# set my output img to zero everywhere except my mask
# output_img = img.copy()
# output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
# output_hsv = img_hsv.copy()
# output_hsv[np.where(mask_green==0)] = 0

area_red = cv2.countNonZero(mask_red[0:int(h/3), 0:w])
area_yellow = cv2.countNonZero(mask_yellow[int(h/3):int(2*h/3), 0:w])
area_green = cv2.countNonZero(mask_green[int(2*h/3):h, 0:w])

color_map = ["RED", "YELLOW", "GREEN"]

print(color_map[np.argmax([area_red, area_yellow, area_green])])

cv2.namedWindow("mask_red", cv2.WINDOW_NORMAL)
cv2.imshow("mask_red", mask_red)
cv2.namedWindow("mask_yellow", cv2.WINDOW_NORMAL)
cv2.imshow("mask_yellow", mask_yellow)
cv2.namedWindow("mask_green", cv2.WINDOW_NORMAL)
cv2.imshow("mask_green", mask_green)
cv2.namedWindow("img_hsv", cv2.WINDOW_NORMAL)
cv2.imshow("img_hsv", img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
