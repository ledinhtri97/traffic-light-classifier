import cv2
import numpy as np

# img = cv2.imread('test/red.jpg')
img = cv2.imread('test/yellow.jpg')
# img = cv2.imread('test/green.jpg')

h, w = img.shape[:2]

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
# output_img = img.copy()
# output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
# output_hsv = img_hsv.copy()
# output_hsv[np.where(mask_green==0)] = 0

area_red = cv2.countNonZero(mask[0:int(h/3), 0:w])
area_yellow = cv2.countNonZero(mask[int(h/3):int(2*h/3), 0:w])
area_green = cv2.countNonZero(mask[int(2*h/3):h, 0:w])

color_map = ["RED", "YELLOW", "GREEN"]

print(color_map[np.argmax([area_red, area_yellow, area_green])])

cv2.namedWindow("i", cv2.WINDOW_NORMAL)
cv2.imshow("i", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()