import cv2
import matplotlib.pyplot as plt
import numpy as np



img=cv2.imread(r'C:\Users\DEEPIKA\Documents\InternCode\bus.jpg',cv2.IMREAD_GRAYSCALE)

hist=cv2.calcHist([img],[0],None,[256],[0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(hist, color='red')
plt.xlim([0, 256])
plt.show()