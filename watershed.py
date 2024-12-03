import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\DEEPIKA\Documents\OpenCVCode\Water_coins.webp')
if img is None:
    print("Error: Image not loaded. Check the file path and file format.")
else:
    cv2.imshow("Original image", img)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold using OTSU
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("Thresholded", thresh)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

cv2.imshow("SureFG", sure_fg)
cv2.imshow("SureBG", sure_bg)
cv2.imshow("Unknown", unknown)

ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]  # Mark boundaries with red color

cv2.imshow("Watershed result", img)
cv2.imwrite('watershed_result.jpg', img) 

cv2.waitKey(0)
cv2.destroyAllWindows()
