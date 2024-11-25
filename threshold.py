
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('basketball.jpg', cv2.IMREAD_GRAYSCALE)

# Apply different thresholding techniques
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
adaptive_threshold=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# Display results using OpenCV (or matplotlib if needed)
# cv2.imshow('Original', image)
# cv2.imshow('Binary', binary)
# cv2.imshow('Binary Inverse', binary_inv)
# cv2.imshow('Truncated', trunc)
# cv2.imshow('To Zero', tozero)
# cv2.imshow('To Zero Inverse', tozero_inv)
# cv2.imshow('adaptive_threshold',adaptive_threshold)

images = [image, binary, binary_inv, trunc, tozero, tozero_inv, adaptive_threshold]
titles = ['Original', 'Binary', 'Binary Inverse', 'Truncated', 'To Zero', 'To Zero Inverse', 'Adaptive Threshold']
plt.figure(figsize=(12, 8))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()


# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
