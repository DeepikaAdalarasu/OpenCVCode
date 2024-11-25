import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('kholi.jpg')

mask=np.zeros(img.shape[:2],np.uint8)

backgroundModel=np.zeros((1,65),np.float64)
foregroundModel=np.zeros((1,65),np.float64)

rectangle=(400,100,1100,700)

cv2.grabCut(img,mask,rectangle,backgroundModel,foregroundModel,3,cv2.GC_INIT_WITH_RECT)

mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')

image_segmented = img * mask2[:, :, np.newaxis]

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB))
plt.axis('off')
 
plt.show()