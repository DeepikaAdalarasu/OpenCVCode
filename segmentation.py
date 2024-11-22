import cv2
from ultralytics import YOLO

model=YOLO('yolov8n-seg.pt')

image_path='dog_bike_car.jpg'

frame=cv2.imread(image_path)

if frame is None:
    print("Error: Frame is not found.")
    exit()

results=model(frame)

annotated_frame=results[0].plot()

# cv2.imshow("YOLO Segmentation",annotated_frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

output_path = 'segmented_output.jpg'
cv2.imwrite(output_path, annotated_frame)
print(f"Segmented image saved to: {output_path}")
