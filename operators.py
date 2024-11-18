
import cv2
import numpy as np

blank=np.zeros((400,400),dtype='uint8')

rectangle=cv2.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle=cv2.circle(blank.copy(),(200,200),200,255,-1)


cv2.imshow('Rectangle',rectangle)
cv2.imshow('Circle',circle)

bitwise_and=cv2.bitwise_and(rectangle,circle)
bitwise_or=cv2.bitwise_or(rectangle,circle)
bitwise_xor=cv2.bitwise_xor(rectangle,circle)
bitwise_not=cv2.bitwise_not(rectangle,circle)

cv2.imshow('bitwise_and',bitwise_and)
cv2.imshow('bitwise_or',bitwise_or)
cv2.imshow('bitwise_xor',bitwise_xor)
cv2.imshow('bitwise_not',bitwise_not)

cv2.waitKey(0)
cv2.destroyAllWindows()