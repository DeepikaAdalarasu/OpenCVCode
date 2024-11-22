
import cv2


cap = cv2.VideoCapture(0)  


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


haar_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    num_faces = len(faces_rect)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv2.putText(frame, f"Faces: {num_faces}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    cv2.imshow('Detected Faces', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
