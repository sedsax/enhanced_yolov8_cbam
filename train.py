from ultralytics import YOLO
import torch
from models.custom_yolov8 import CustomYOLOv8

def main():
    # 1. YOLOv8n modelini temel al
    model = YOLO('yolov8n.pt')
    
    # 4. Eğitim yapılandırması
    model.train(
        data='data.yaml',  # Veri yapılandırma dosyası
        epochs=15,  # Epoch sayısı
        imgsz=640,  # Görüntü boyutu
        batch=16,  # Batch boyutu
        device=0,  # GPU indeksi
    )

if __name__ == "__main__":
    main()