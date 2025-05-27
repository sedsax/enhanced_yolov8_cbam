"""
YOLOv8 model mimarisini detaylı olarak analiz et ve CBAM entegrasyonunu test et.
Bu script, YOLOv8 modelini detaylı inceleyerek en etkili CBAM entegrasyon stratejisini bulur.
"""

import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from improved_yolov8_cbam import ImprovedYOLOv8WithCBAM
from PIL import Image
import seaborn as sns

# Temel dizin
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml') 
MODEL_PATH = os.path.join(ROOT_DIR, 'yolov8n.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'model_analysis')

# Çıktı klasörü yoksa oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_module_structure():
    """YOLOv8 model yapısını görselleştir"""
    print("\nYOLOv8 modül yapısını görselleştirme...")
    
    # Modeli yükle
    model = YOLO(MODEL_PATH)
    
    # Modül isimleri ve türleri
    module_info = []
    for name, module in model.model.named_modules():
        if len(name) > 0:  # Boş isimli modülleri atla
            module_info.append((name, type(module).__name__))
    
    # Modül yapısını görselleştir
    plt.figure(figsize=(20, 12))
    
    # Modül hiyerarşisini göster
    max_depth = max(len(name.split('.')) for name, _ in module_info)
    y_pos = len(module_info)
    
    for i, (name, module_type) in enumerate(module_info):
        parts = name.split('.')
        depth = len(parts)
        
        # Nokta sayısına göre renklendirme
        color_intensity = 0.6 - (depth / max_depth) * 0.5
        color = (0, color_intensity, 0.8)
        
        # Modülü çiz
        plt.barh(y_pos - i, depth, color=color, height=0.8)
        
        # İsmi ve türü göster
        plt.text(depth + 0.1, y_pos - i, f"{name} ({module_type})", 
                 ha='left', va='center', fontsize=8)
    
    plt.title('YOLOv8 Model Yapısı')
    plt.xlabel('Modül Derinliği')
    plt.ylabel('Modüller')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'yolov8_module_structure.png'), dpi=300)

def get_test_image():
    """Test için örnek bir görüntü getir"""
    test_images_dir = os.path.join(ROOT_DIR, 'dataset', 'test', 'images')
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        test_image_path = os.path.join(test_images_dir, image_files[0])
        print(f"Test için örnek görüntü: {test_image_path}")
        return test_image_path
    else:
        raise FileNotFoundError("Test klasöründe görüntü bulunamadı!")

def visualize_activation_maps(image_path):
    """
    Standart YOLOv8 ve CBAM entegre edilmiş YOLOv8'in aktivasyon haritalarını karşılaştır
    """
    print("\nAktivasyon haritaları oluşturuluyor...")
    
    # Orijinal görüntüyü yükle ve boyutlandır
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))
    
    # Modelleri yükle
    standard_model = YOLO(MODEL_PATH)
    cbam_model = ImprovedYOLOv8WithCBAM(MODEL_PATH)
    
    # Görüntüyü işle ve tahminleri al
    standard_results = standard_model.predict(img_resized)
    cbam_results = cbam_model.predict(img_resized)
    
    # Görüntüleri görselleştir
    plt.figure(figsize=(15, 10))
    
    # Orijinal görüntü
    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    # Standart model tahminleri
    plt.subplot(1, 3, 2)
    plt.imshow(img_resized)
    for box in standard_results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                          fill=False, edgecolor='blue', linewidth=2))
    plt.title("Standart YOLOv8")
    plt.axis('off')
    
    # CBAM modeli tahminleri
    plt.subplot(1, 3, 3)
    plt.imshow(img_resized)
    for box in cbam_results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                          fill=False, edgecolor='red', linewidth=2))
    plt.title("CBAM-YOLOv8")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=300)
    print(f"Model karşılaştırma görselleştirmesi kaydedildi: {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")

def analyze_yolov8_architecture():
    """YOLOv8 mimarisini kapsamlı şekilde analiz et"""
    print("\n" + "="*80)
    print("YOLOv8 MİMARİSİ DETAYLI ANALİZ")
    print("="*80)
    
    # Modeli yükle
    model = YOLO(MODEL_PATH)
    nn_model = model.model
    
    # Model genel bilgileri
    print("\n1. GENEL MODEL BİLGİLERİ")
    print("-"*40)
    total_params = sum(p.numel() for p in nn_model.parameters())
    print(f"Toplam parametreler: {total_params:,}")
    print(f"Model boyutu: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    # Ana modül yapısı
    print("\n2. ANA MODÜL YAPISI")
    print("-"*40)
    for name, module in nn_model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name} ({type(module).__name__}): {params:,} parametre ({params/total_params*100:.2f}%)")
    
    # Backbone analizi
    print("\n3. BACKBONE ANALİZİ")
    print("-"*40)
    if hasattr(nn_model, 'backbone'):
        backbone = nn_model.backbone
        for name, module in backbone.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"{name} ({type(module).__name__}): {params:,} parametre")
            
            # C2f blokları için detaylı bilgi
            if 'C2f' in name:
                print(f"  - Bu bir CBAM eklemek için ideal yerdir")
    
    # Head analizi
    print("\n4. DETECTION HEAD ANALİZİ")
    print("-"*40)
    if hasattr(nn_model, 'head'):
        head = nn_model.head
        for name, module in head.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"{name} ({type(module).__name__}): {params:,} parametre")
    
    # CBAM entegre edilecek en iyi konumları belirle
    print("\n5. CBAM İÇİN OPTİMAL KONUMLAR")
    print("-"*40)
    
    # Tüm modül isimlerini ve tiplerini listele
    target_types = []
    for name, module in nn_model.named_modules():
        module_type = type(module).__name__
        parts = name.split('.')
        
        # Hedef modül türlerini belirle
        if len(parts) <= 3 and not any(x in module_type for x in ['Conv2d', 'BatchNorm2d', 'SiLU']):
            if (('backbone.C2f' in name) or 
                ('head' in name and any(x in name for x in ['cv2', 'cv3']))):
                target_types.append((name, module_type))
    
    # Optimize edilmiş hedefleri listele
    print("CBAM için önerilen konumlar:")
    for name, module_type in target_types:
        print(f"- {name} ({module_type})")
    
    # Kanal sayılarını belirle
    cbam_locations = []
    for name, _ in target_types:
        try:
            parts = name.split('.')
            current = nn_model
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            
            # Kanal sayısı belirlenebilir mi?
            channels = None
            
            # Alt modüllerde konvolüsyon katmanı ara
            for subname, submodule in current.named_modules():
                if isinstance(submodule, torch.nn.Conv2d) and submodule.out_channels >= 64:
                    channels = submodule.out_channels
                    break
            
            if channels:
                cbam_locations.append((name, channels))
                print(f"  Kanal sayısı: {channels}")
        except Exception as e:
            print(f"  Hata: {e}")
    
    print(f"\nÖnerilen CBAM konumları: {len(cbam_locations)}")
    for name, channels in cbam_locations:
        print(f"- {name}: {channels} kanal")
    
    return cbam_locations

def main():
    print("YOLOv8 Model Mimarisi ve CBAM Entegrasyonu Analizi Başlatılıyor...")
    
    # Modeli analiz et
    cbam_locations = analyze_yolov8_architecture()
    
    # Model yapısını görselleştir
    visualize_module_structure()
    
    # Test görüntüsü al
    try:
        test_image = get_test_image()
        
        # Aktivasyon haritalarını görselleştir
        visualize_activation_maps(test_image)
    except Exception as e:
        print(f"Görüntü işleme hatası: {e}")
    
    print("\nAnaliz Tamamlandı!")
    print(f"Çıktılar '{OUTPUT_DIR}' klasörüne kaydedildi.")
    
    print("\nYOLOv8 mimarisinin incelemesinden sonra elde edilen bulgular:")
    print("1. Backbone'daki C2f blokları CBAM eklemek için ideal konumlardır")
    print("2. Detection Head'in cv2 ve cv3 bileşenleri de CBAM'dan yararlanabilir")
    print("3. Bu analizi kullanarak 'improved_yolov8_cbam.py' üzerinden CBAM entegrasyonunu yapabilirsiniz")
    print(f"4. Önerilen CBAM konumları: {len(cbam_locations)} adet")

if __name__ == "__main__":
    main()
