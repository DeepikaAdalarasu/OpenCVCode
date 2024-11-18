import os
import cv2

img=cv2.imread(r"C:\Users\DEEPIKA\Documents\InternCode\cat.jpg")

k_size=7
sigmaX = 1
img_blur=cv2.blur(img,(k_size,k_size))
gaussian_blur=cv2.GaussianBlur(img,(k_size,k_size),sigmaX)
median_blur=cv2.medianBlur(img,(k_size))

cv2.imshow('img_blur',img_blur)
cv2.imshow('gaussian_blur',gaussian_blur)
cv2.imshow('medain_blur',median_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()