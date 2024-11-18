import cv2
img=cv2.imread(r"C:\Users\DEEPIKA\Documents\InternCode\cat.jpg")

print(img.shape)

crop_img=img[100:480,220:560]
resized_img=cv2.resize(img,(220,270))
cv2.imshow('img',img)
cv2.imshow('crop_img',crop_img)
cv2.imshow('resized_img',resized_img)
cv2.waitKey(0)