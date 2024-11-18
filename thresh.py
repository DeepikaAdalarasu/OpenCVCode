import cv2
import numpy as np


image = cv2.imread(r'C:\Users\DEEPIKA\Documents\InternCode\bus.jpg', cv2.IMREAD_GRAYSCALE)


_, binary_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


_, binary_inv_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

mean_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

# Apply Adaptive Gaussian Thresholding
gaussian_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)


cv2.imshow('Original Image', image)
cv2.imshow('Binary Threshold', binary_thresh)
cv2.imshow('Inverse Binary Threshold', binary_inv_thresh)
cv2.imshow('Mean Adaptive Thresholding', mean_thresh)
cv2.imshow('Gaussian Adaptive Thresholding', gaussian_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
