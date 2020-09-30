import cv2
import matplotlib.pyplot as plt

img_path = 'data/seattle.jpg'

bgr_image = cv2.imread(img_path)
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

gray_image = gray_image.astype("float32")/255
plt.imshow(gray_image, cmap='gray')
plt.show()