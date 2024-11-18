import cv2
import os
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\DEEPIKA\Documents\InternCode\cat.jpg")
rotate_180=cv2.rotate(img,cv2.ROTATE_180)
rotate_90=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
rotate_90_COUNTERCLOCKWISE=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('rotate_180',rotate_180)
cv2.imshow('rotate_90',rotate_90)
cv2.imshow('rotate_90_COUNTERCLOCKWISE',rotate_90_COUNTERCLOCKWISE)
cv2.waitKey(0)