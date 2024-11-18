import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread(r'C:\Users\DEEPIKA\Documents\InternCode\phone.webp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)
points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])
points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])
M = cv2.getPerspectiveTransform(points_A, points_B)
warped = cv2.warpPerspective(image, M, (420,594))
plt.subplot(1, 2, 2)
plt.title("warpPerspective")
plt.imshow(warped)
plt.show()