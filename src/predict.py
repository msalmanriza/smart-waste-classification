import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("models/best_model.keras")
detector = YOLO("models/yolov8n.pt")

class_names = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

def predict_image(image):

    image = image.convert("RGB")

    frame = np.array(image)

    results = detector(frame, verbose=False)
    print(f"Detected Objects: {len(results[0].boxes)}")

    if len(results[0].boxes) > 0:

        box = results[0].boxes[0]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            frame = crop
            print("Yolo Crop Success")

    image = Image.fromarray(frame)
    image = image.resize((224, 224))

    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    pred = model.predict(arr, verbose=0)
    idx = np.argmax(pred[0])

    return {
        "class": class_names[idx],
        "confidence": float(pred[0][idx])
    }