import cv2
import numpy as np

# img = cv2.imread('traffic-light/red.jpg')
# img = cv2.imread('traffic-light/red_1/1162.jpg')

img = cv2.imread('traffic-light/yellow.jpg')
# img = cv2.imread('traffic-light/yellow_1/78.jpg')

# img = cv2.imread('traffic-light/green.jpg')
# img = cv2.imread('traffic-light/green_1/97.jpg')

img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask_red0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask_red1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask_red = mask_red0+mask_red1

# lower mask (36-70)
lower_green = np.array([45,50,50])
upper_green = np.array([95,255,255])
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

# lower mask (36-70)
lower_yellow = np.array([12,50,50])
upper_yellow = np.array([36,255,255])
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

mask = mask_red + mask_yellow + mask_green

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0
h, w = output_img.shape[:2]

# or your HSV image, which I *believe* is what you want
# output_hsv = img_hsv.copy()
# output_hsv[np.where(mask_green==0)] = 0

area_red = cv2.countNonZero(cv2.cvtColor(output_img[0:int(h/3), 0:w], cv2.COLOR_BGR2GRAY))
area_yellow = cv2.countNonZero(cv2.cvtColor(output_img[int(h/3):int(2*h/3), 0:w], cv2.COLOR_BGR2GRAY))
area_green = cv2.countNonZero(cv2.cvtColor(output_img[int(2*h/3):h, 0:w], cv2.COLOR_BGR2GRAY))

color_map = ["RED", "YELLOW", "GREEN"]

print(color_map[np.argmax([area_red, area_yellow, area_green])])

cv2.namedWindow("i", cv2.WINDOW_NORMAL)
cv2.imshow("i", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()