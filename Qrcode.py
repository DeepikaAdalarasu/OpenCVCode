import cv2
import webbrowser

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

while True:
    _, img = cap.read()

    data, bbox, _ = detector.detectAndDecode(img)

    if data:
        a = data
        break  
    cv2.imshow("QRCODEscanner", img)
    if cv2.waitKey(1) == ord("q"):  
        break

if data:  
    webbrowser.open(str(a))

cap.release()
cv2.destroyAllWindows()