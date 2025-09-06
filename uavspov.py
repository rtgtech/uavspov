import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

input_path = "input_video.mp4"  
cap = cv2.VideoCapture(input_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False, device='cuda')[0]  
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        if cls_id == 0:  
            color = (255, 255, 255) 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            cv2.drawMarker(frame, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        elif cls_id == 2:  
            color = (0, 0, 255)  
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        elif cls_id == 3:  
            color = (255, 0, 0)  
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
