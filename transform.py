import cv2
import numpy as np
img=cv2.imread(r"C:\Users\DEEPIKA\Documents\InternCode\cat.jpg")
tx,ty=70,90
translation_matrix=np.float32([[1,0,tx],[0,1,ty]])
shifted_img=cv2.warpAffine(img,translation_matrix,(img.shape[1],img.shape[0]))
cv2.imshow('shifted_img',shifted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()