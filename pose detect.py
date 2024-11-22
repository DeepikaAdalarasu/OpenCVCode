# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# model = YOLO('yolov8n-pose.pt')

# img_path=r"C:\Users\DEEPIKA\Documents\OpenCVCode\pose.jpeg"

# frame=cv2.imread(img_path)

# if frame is None:
#     print("Error: Frame is not found.")
#     exit()

# results=model(frame)

# annotated_frame=results[0].plot()

# cv2.imshow("Pose Detection",annotated_frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('yolov8n-pose.pt')

img_path = r'C:\Users\DEEPIKA\Documents\OpenCVCode\basketball.jpg'

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image. Please check the path.")
    exit()

results = model(image)

annotated_image = results[0].plot()

annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.imshow(annotated_image_rgb)
plt.axis('off')  
plt.title("YOLOv8 Pose Estimation")
plt.show()
