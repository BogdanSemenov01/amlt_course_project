import cv2
from ultralytics import YOLO

# Labels 
objects = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]


# Connect via IP Cam
cap = cv2.VideoCapture('https://192.168.0.14:8080/video')

# Connect via ADB
# cap = ADBVideoCapture(False)
# cap.open()

model = YOLO('weights/yolov10n.onnx')

while True:
    ret, frame = cap.read()
    
    results = model(frame, verbose=False)[0]

    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, f'{objects[int(label.detach().item())]} {confidence.detach().item():.2f}', (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Android Cam", frame)

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
