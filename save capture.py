# 

import cv2
import os
import time

save_folder = r'C:\Users\DEEPIKA\Documents\CapturedImages'

cap = cv2.VideoCapture(0)


for i in range(1, 5):
    
    ret, frame = cap.read()
  
    filename = os.path.join(save_folder, f'img_{i}.jpg')
    cv2.imwrite(filename, frame)
    print(f"Img {i} saved as {filename}")
    time.sleep(3)


cap.release()
cv2.destroyAllWindows()
