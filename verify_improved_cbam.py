"""
Bu script YOLOv8 modeline düzgün şekilde CBAM entegre edildiğini doğrular
ve iyileştirilmiş ImprovedYOLOv8WithCBAM sınıfını kullanır.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from improved_yolov8_cbam import ImprovedYOLOv8WithCBAM
from PIL import Image

# Temel dizin
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml')
MODEL_PATH = os.path.join(ROOT_DIR, 'yolov8n.pt')
BASIC_MODEL_PATH = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_basic', 'weights', 'best.pt')
CBAM_MODEL_PATH = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_cbam', 'weights', 'best.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'cbam_verification')

# Çıktı dizinini oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sample_image():
    """Test görüntüsü getir"""
    test_dir = os.path.join(ROOT_DIR, 'dataset', 'test', 'images')
    images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
             if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        raise FileNotFoundError(f"Test görüntüsü bulunamadı: {test_dir}")
    
    # İlk görüntüyü kullan
    sample_img = images[0]
    print(f"Test için örnek görüntü: {sample_img}")
    return sample_img

def create_improved_cbam_model():
    """Geliştirilmiş CBAM entegrasyonlu model oluştur"""
    print("\nGeliştirilmiş CBAM entegrasyonlu YOLOv8 modeli oluşturuluyor...")
    
    # Geliştirilmiş CBAM modeli oluştur
    cbam_model = ImprovedYOLOv8WithCBAM(MODEL_PATH)
    
    # Model parametrelerini kontrol et
    original_params = sum(p.numel() for p in YOLO(MODEL_PATH).model.parameters())
    cbam_params = sum(p.numel() for p in cbam_model.nn_model.parameters())
    
    print("\nModel Özeti:")
    print(f"Orijinal YOLOv8 parametreleri: {original_params:,}")
    print(f"CBAM iyileştirilmiş parametreler: {cbam_params:,}")
    print(f"Ek parametreler: {cbam_params - original_params:,}")
    
    return cbam_model

def check_trained_models():
    """Eğitilmiş modelleri kontrol et"""
    print("\nEğitilmiş modeller kontrol ediliyor:")
    standard_model_exists = os.path.exists(BASIC_MODEL_PATH)
    cbam_model_exists = os.path.exists(CBAM_MODEL_PATH)
    
    print(f"Standart model: {'MEVCUT' if standard_model_exists else 'YOK'}")
    print(f"CBAM model: {'MEVCUT' if cbam_model_exists else 'YOK'}")
    
    return standard_model_exists, cbam_model_exists

def visualize_models(image_path):
    """Standart ve CBAM modellerini görselleştir"""
    print("\nEğitilmiş modeller görselleştiriliyor...")
    
    # Modelleri yükle
    standard_model = YOLO(BASIC_MODEL_PATH)
    cbam_model = YOLO(CBAM_MODEL_PATH)
    
    # Tahminleri al
    standard_results = standard_model.predict(image_path, verbose=True)
    cbam_results = cbam_model.predict(image_path, verbose=True)
    
    # Karşılaştırma görselleştirmesi oluştur
    img = Image.open(image_path)
    img = np.array(img)
    
    # Görselleştirme
    plt.figure(figsize=(15, 5))
    
    # Orijinal görüntü
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    # Standart model sonuçları
    plt.subplot(1, 3, 2)
    for r in standard_results:
        boxes = r.boxes
        plt.imshow(img)
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, edgecolor='blue', linewidth=2))
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            plt.text(x1, y1-5, f'Acne: {conf:.2f}', color='blue', 
                    fontsize=10, backgroundcolor='white')
    plt.title("YOLOv8 Standart")
    plt.axis('off')
    
    # CBAM model sonuçları
    plt.subplot(1, 3, 3)
    for r in cbam_results:
        boxes = r.boxes
        plt.imshow(img)
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, edgecolor='red', linewidth=2))
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            plt.text(x1, y1-5, f'Acne: {conf:.2f}', color='red', 
                    fontsize=10, backgroundcolor='white')
    plt.title("YOLOv8 + CBAM")
    plt.axis('off')
    
    # Kaydedilecek dosya yolu
    output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Model karşılaştırma görselleştirmesi kaydedildi: {output_path}")

def main():
    print("CBAM doğrulama işlemi başlatılıyor...")
    
    # Test görüntüsü al
    sample_img = get_sample_image()
    print(f"Test için örnek görüntü: {sample_img}")
    
    # Geliştirilmiş CBAM modelini oluştur
    cbam_model = create_improved_cbam_model()
    
    # Eğitilmiş modelleri kontrol et
    standard_exists, cbam_exists = check_trained_models()
    
    # Eğitilmiş modelleri görselleştir
    if standard_exists and cbam_exists:
        visualize_models(sample_img)
    
    print("\nYOLOv8 vs YOLOv8-CBAM Doğrulaması Tamamlandı!")
    print(f"Görselleştirmeleri '{OUTPUT_DIR}' dizininde inceleyebilirsiniz.")

if __name__ == "__main__":
    main()
