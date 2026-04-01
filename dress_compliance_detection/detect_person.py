# person_detector.py

import cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", min_width=50, min_height=100, conf_thresh=0.5):
        """
        min_width, min_height: minimum box size to consider a person
        conf_thresh: minimum confidence to accept detection
        """
        print("[INFO] Loading YOLOv8 model for person detection...")
        self.model = YOLO(model_path)
        self.min_width = min_width
        self.min_height = min_height
        self.conf_thresh = conf_thresh

    def detect_persons(self, frame):
        """
        Detect persons in the frame
        Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
        """
        results = self.model(frame)
        person_boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf >= self.conf_thresh:  # class 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    # Filter tiny boxes
                    if w >= self.min_width and h >= self.min_height:
                        person_boxes.append((x1, y1, x2, y2))

        return person_boxes

# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    detector = PersonDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect_persons(frame)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
