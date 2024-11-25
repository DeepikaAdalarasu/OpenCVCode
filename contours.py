import cv2
import matplotlib.pyplot as plt

image = cv2.imread('dog_bike_car.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

# output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.imshow(output)
plt.title('Contours')
plt.axis('off')
plt.show()
