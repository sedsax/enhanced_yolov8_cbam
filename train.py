import sys
sys.path.insert(0, './ultralytics')
from ultralytics import YOLO

if __name__ == "__main__":
    # Model ve veri yolu
    model = YOLO('yolov8n.yaml')  # veya başka bir yolov8 ağırlığı

    # Eğitim parametreleri D:\Projects\object_detect_v8\ultralytics\ultralytics\cfg\models\v8\yolov8-ghost-p2.yaml
    model.train(
        data='dataset/data.yaml',  # Eğitim verisi için YAML dosyası
        epochs=15,           # Eğitim epoch sayısı
        imgsz=640,           # Görüntü boyutu
        batch=4,            # Batch size
    )