import cv2
img=cv2.imread(r"C:\Users\DEEPIKA\Documents\InternCode\cat.jpg")

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


cv2.imshow('img_hsv',img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()