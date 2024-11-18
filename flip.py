import cv2
img=cv2.imread(r"C:\Users\DEEPIKA\Documents\InternCode\cat.jpg")
flip_vertical=cv2.flip(img,0)
flip_horizontal=cv2.flip(img,1)
flip_both=cv2.flip(img,-1)

cv2.imshow('flip_vertical',flip_vertical)
cv2.imshow('flip_horizontal',flip_horizontal)
cv2.imshow('flip_both',flip_both)

cv2.waitKey(0)
cv2.destroyAllWindows()

