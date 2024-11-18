import cv2


img=cv2.imread(r'C:\Users\DEEPIKA\Documents\InternCode\group.webp')
# cv2.imshow('img',img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

haar_cascade=cv2.CascadeClassifier('haarcascade_face.xml')
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
print(f'number of faces found={len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)


cv2.imshow('Detected faces',img)

cv2.waitKey(0)
cv2.destroyAllWindows()