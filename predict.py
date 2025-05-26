from models.custom_yolov8 import CustomYOLOv8
from PIL import Image
import torch
import numpy as np

def main():
    # 1. Eğitilmiş modeli yükle
    model = CustomYOLOv8()
    model.load('path/to/saved/weights.pt')
    
    # 2. Görüntü yükle
    img = Image.open('path/to/test_image.jpg')
    
    # 3. Tahmin yap
    results = model.predict(img, conf=0.25)
    
    # 4. Sonuçları işle
    for result in results:
        boxes = result.boxes  # Nesne kutuları
        print(f"Detected {len(boxes)} objects")
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Kutu koordinatları (x1, y1, x2, y2 formatında)
            conf = box.conf[0]  # Güven skoru
            cls = int(box.cls[0])  # Sınıf indeksi
            print(f"Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), Confidence: {conf:.4f}, Class: {cls}")

if __name__ == "__main__":
    main()