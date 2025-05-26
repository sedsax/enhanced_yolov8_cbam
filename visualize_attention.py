from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_attention(std_model_path, cbam_model_path, image_path, output_dir='visualizations'):
    """Standard ve CBAM modeli arasındaki farkları görselleştirir"""
    # Model yükleme
    std_model = YOLO(std_model_path)
    cbam_model = YOLO(cbam_model_path)
    
    # Çıktı dizini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Tahminler
    std_results = std_model(img.copy())
    cbam_results = cbam_model(img.copy())
    
    # Görselleştirme
    plt.figure(figsize=(16, 8))
    
    # Orijinal görüntü
    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(img)
    plt.axis('off')
    
    # Standart model tahminleri
    plt.subplot(1, 3, 2)
    plt.title(f"Standart YOLOv8: {len(std_results[0].boxes)} nesne")
    std_img = img.copy()
    for box in std_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        
        # Nesne çerçevesi
        cv2.rectangle(std_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Etiket
        label = f"{std_results[0].names[cls]} {conf:.2f}"
        cv2.putText(std_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.imshow(std_img)
    plt.axis('off')
    
    # CBAM model tahminleri
    plt.subplot(1, 3, 3)
    plt.title(f"CBAM YOLOv8: {len(cbam_results[0].boxes)} nesne")
    cbam_img = img.copy()
    for box in cbam_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        
        # Nesne çerçevesi
        cv2.rectangle(cbam_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Etiket
        label = f"{cbam_results[0].names[cls]} {conf:.2f}"
        cv2.putText(cbam_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    plt.imshow(cbam_img)
    plt.axis('off')
    
    # Kaydet
    img_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f"comparison_{img_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    
    print(f"Görselleştirme kaydedildi: {save_path}")
    return save_path

def compare_multiple_images(std_model_path, cbam_model_path, image_dir, output_dir='visualizations'):
    """Birden fazla test görüntüsü için karşılaştırma yapar"""
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    for img_path in image_files[:10]:  # İlk 10 görüntü
        print(f"İşleniyor: {img_path}")
        save_path = visualize_attention(std_model_path, cbam_model_path, img_path, output_dir)
        results.append(save_path)
    
    return results

# Örnek kullanım
if __name__ == "__main__":
    # Modellerin yolu
    std_model = "runs/detect/train5/weights/best.pt"  # Standart model yolu
    cbam_model = "runs/detect/cbam_final2/weights/best.pt"  # CBAM model yolu
    
    # Tek görüntü analizi
    visualize_attention(std_model, cbam_model, "dataset/valid/images/acne-2_jpeg.rf.f4ec613e23dd372f3232c4cd68424250.jpg")
    
    # Çoklu görüntü analizi
    # compare_multiple_images(std_model, cbam_model, "dataset/valid/images")