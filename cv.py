# import os
# import cv2
# image_path=os.path.join('.','image_data','lib2.jpeg')
# print(image_path) 
# img=cv2.imread(image_path)
# cv2.imwrite(os.path.join('.','image_data','lib2_out.jpeg'),img)
# cv2.imshow('image',img)
# cv2.waitKey(5000)
# # roi = img[100 : 500, 200 : 700]
# # cv2.imshow("ROI", roi)
# # cv2.waitKey(5000)
# # output = img.copy()

# # # Using the rectangle() function to create a rectangle.
# # rectangle = cv2.rectangle(output, (1500, 900),
# #                         (600, 400), (255, 0, 0), 2)
# # text = cv2.putText(output, 'OpenCV Demo', (500, 550),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2)
# #print(img)
# #print(img.shape)  #shows height,width,no.of channels
# import os
# import cv2
# video_path=r'C:\Users\DEEPIKA\Documents\IdeassionInternship\image_data\car_moving.mp4'
# video=cv2.VideoCapture(video_path)

# ret = True
# while ret:
#     ret,frame=video.read()
#     if ret:
#         cv2.imshow('frame',frame)
#         cv2.waitKey(40)

# video.release()
# cv2.destroyAllWindows()

# # import os
# # import cv2

# # video_path = r'C:\Users\DEEPIKA\Documents\IdeassionInternship\image_data\car_moving.mp4'
# # video = cv2.VideoCapture(video_path)
# # print("Video path:", video_path)  # Check the resolved path
# # print("File exists:", os.path.exists(video_path)) 

# # # Check if video loaded correctly
# # if not video.isOpened():
# #     print("Error: Could not open video.")
# # else:
# #     while True:
# #         ret, frame = video.read()
# #         if not ret:
# #             print("End of video or error loading frame.")
# #             break
# #         cv2.imshow('frame', frame)
        
# #         # Delay to control playback speed; adjust as needed
# #         if cv2.waitKey(40) & 0xFF == ord('q'):  # Press 'q' to quit
# #             break

# # video.release()
# # cv2.destroyAllWindows()
# import cv2
# webcam=cv2.VideoCapture(0)
# while True:
#     ret,frame=webcam.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break
# webcam.release()
# import os
# import cv2
# img = cv2.imread('image_data\lib2.jpeg')
# if img is None:
#     print("Error: Could not load image.")
# img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',img)
# cv2.imshow('img_rgb',img_rgb)
# cv2.waitKey(0)
# import os
# import cv2
# img=cv2.imread(r'image_data\bird.jpg')
# if img is None:
#      print("Error: Could not load image.")
# cv2.imshow('img',img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret,thresh=cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
# thresh=cv2.blur(thresh,(10,10))
# ret,thresh=cv2.threshold(thresh,80,255,cv2.THRESH_BINARY)
# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import numpy as np
# import os
# import cv2
# img=cv2.imread(r'image_data\basketball.jpg')
# img_edge=cv2.Canny(img,400,600)
# kernel = np.ones((5, 5), np.uint8)
# img_edge_d=cv2.dilate(img_edge,kernel)
# cv2.imshow('img',img)
# cv2.imshow('img_edge',img_edge)
# cv2.imshow('img_edge_d',img_edge_d)
# cv2.waitKey(0)

# import os
# import cv2
# img=cv2.imread(r'image_data\whiteboard.jpg')
# cv2.line(img,(100,150),(300,450),(0,255,0),3)
# cv2.rectangle(img,(200,350),(450,600),(0,0,255),7)
# cv2.circle(img,(100,200),20,(255,0,0),3)
# cv2.putText(img,'Hey You!',(200,450),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
# print(img.shape)
# cv2.imshow('img',img)
# cv2.waitKey(0)
from util import get_limits
from PIL import Image

import cv2
import numpy as np
blue=[0,255,0]
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    hsvImage=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerlimit,upperlimit=get_limits(color=blue)
    mask=cv2.inRange(hsvImage,lowerlimit,upperlimit)
    mask_=Image.fromarray(mask)
    bbox=mask_.getbbox()
    if bbox is not None:
        x1,y1,x2,y2=bbox
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5)
    cv2.imshow('frame',frame)



#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import matplotlib.pyplot as plt
# img=cv2.imread('image_data/bird.jpg')
# gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# blur_img=cv2.GaussianBlur(gray_img,(5,5),0)
# edges=cv2.Canny(blur_img,threshold1=50,threshold2=150)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# cv2.imshow("Original Image", img)
# cv2.imshow("Edges", edges)

# # Wait for a key press to close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# img_path='image_data/basketball.jpg'
# img=cv2.imread(img_path)
# mp_face_detection=mp.solutions.face_detection
# with mp_face_detection.FaceDetection(min_detection_confidence=0,model_selection=0) as face_detection:
#     img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     out=face_detection.process(img_rgb)
# for detection in out.detection:

# cv2.imshow("Face Detection", img)
        
#         # Wait until a key is pressed to close the image window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

