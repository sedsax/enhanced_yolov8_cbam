import torch
from ultralytics import YOLO
from models.yolov8_cbam import YOLOv8_CBAM
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_activation_maps(model, image_path, layer_idx=6):
    """Belirli bir katmandan aktivasyon haritalarını alır"""
    activations = []
    
    # Hook fonksiyonu
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    # Hook ekle
    if hasattr(model.model, 'model'):
        layer = model.model.model[layer_idx]
        hook = layer.register_forward_hook(hook_fn)
    
    # Tahmin yap
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(image_path)
    
    # Hook'u kaldır
    hook.remove()
    
    return activations, img, results

def visualize_comparison(org_model_path, cbam_model_path, image_path, layer_idx=6):
    """İki model arasındaki aktivasyon haritalarını karşılaştırır"""
    # Modelleri yükle
    org_model = YOLO(org_model_path)
    cbam_model = YOLOv8_CBAM(cbam_model_path)
    
    # Aktivasyonları al
    org_acts, img, org_results = get_activation_maps(org_model, image_path, layer_idx)
    cbam_acts, _, cbam_results = get_activation_maps(cbam_model, image_path, layer_idx)
    
    # Görselleştir
    plt.figure(figsize=(18, 10))
    
    # Orijinal görüntü
    plt.subplot(2, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(img)
    
    # Tahmin sonuçları
    plt.subplot(2, 3, 2)
    plt.title("YOLOv8 Tahminleri")
    plt.imshow(img)
    for box in org_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
    
    plt.subplot(2, 3, 3)
    plt.title("YOLOv8+CBAM Tahminleri")
    plt.imshow(img)
    for box in cbam_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='blue', linewidth=2))
    
    # Aktivasyon haritaları
    if org_acts and cbam_acts:
        # Ortalama aktivasyon haritası
        org_act = np.mean(org_acts[0][0], axis=0)
        cbam_act = np.mean(cbam_acts[0][0], axis=0)
        
        plt.subplot(2, 3, 4)
        plt.title("YOLOv8 Aktivasyon Haritası")
        plt.imshow(org_act, cmap='viridis')
        
        plt.subplot(2, 3, 5)
        plt.title("YOLOv8+CBAM Aktivasyon Haritası")
        plt.imshow(cbam_act, cmap='viridis')
        
        plt.subplot(2, 3, 6)
        plt.title("Fark Haritası (CBAM Etkisi)")
        plt.imshow(cbam_act - org_act, cmap='coolwarm')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

# Test görüntüsü üzerinde karşılaştırma yap
visualize_comparison(
    'runs/detect/train4/weights/best.pt',  # Orijinal model yolu
    'runs/detect/custom_model2/weights/best.pt',  # CBAM model yolu
    'path/to/test_image.jpg'  # Test görüntüsü
)