import ultralytics
from ultralytics import YOLO

YOLO(model='yolov8n.pt', task='detect')

def detect(path: str):
    results = YOLO('yolov8n.pt').predict(source=path, save=True, save_txt=True, show=True)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]
            detections.append({'label': label, 'confidence': conf})
    return detections