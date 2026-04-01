from detect_person import PersonDetector
import cv2

detector = PersonDetector()
# frame = cv2.imread("C:\\Users\\jsril\\Spyderprojects\\DressAttireDetection\\dataset\\casual\\51OQt3Y5A7LSX679_.jpg")

#ball image
frame = cv2.imread("C:\\Users\\jsril\\Downloads\\ball.jpg")
boxes = detector.detect_persons(frame)

for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("Persons", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
