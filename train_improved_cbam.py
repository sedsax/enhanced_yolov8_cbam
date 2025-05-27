"""
Geliştirilmiş CBAM entegrasyonlu YOLOv8 modelini eğitme scripti
"""

from improved_yolov8_cbam import ImprovedYOLOv8WithCBAM
import os
import argparse
import time

# Temel dizin
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml')
MODEL_PATH = os.path.join(ROOT_DIR, 'yolov8n.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_improved_cbam')

def train_improved_cbam_model(epochs=100, batch_size=16, patience=20, device='0'):
    """
    Geliştirilmiş CBAM entegrasyonlu YOLOv8 modelini eğit
    
    Args:
        epochs: Eğitim devir sayısı
        batch_size: Batch boyutu
        patience: Erken durdurma sabır süresi
        device: Kullanılacak cihaz (CPU=cpu, GPU=0,1,2...)
    """
    print("Geliştirilmiş YOLOv8+CBAM modeli eğitiliyor...")
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # Geliştirilmiş CBAM modelini oluştur
    print("\n1. Model oluşturuluyor...")
    model = ImprovedYOLOv8WithCBAM(MODEL_PATH, debug=True)
    
    # Modeli eğit
    print("\n2. Model eğitimi başlatılıyor...")
    print(f"Eğitim parametreleri: {epochs} devir, {batch_size} batch boyutu")
    
    model.train(
        data=DATA_YAML,          # Veri konfigürasyonu
        epochs=epochs,           # Devir sayısı  
        imgsz=640,               # Görüntü boyutu
        batch=batch_size,        # Batch boyutu
        name='train_improved_cbam', # Çalıştırma adı
        patience=patience,        # Erken durdurma sabır süresi
        save=True,               # Kontrol noktalarını kaydet
        device=device,           # Kullanılacak cihaz
        verbose=True,            # Detaylı çıktı
        exist_ok=True            # Varolan klasörün üzerine yaz
    )
    
    # Eğitim süresini hesapla
    training_time = (time.time() - start_time) / 3600  # saat cinsinden
    
    print(f"\nEğitim tamamlandı. Sonuçlar {OUTPUT_DIR} konumuna kaydedildi.")
    print(f"Toplam eğitim süresi: {training_time:.3f} saat")

def main():
    # Komut satırı argümanlarını yapılandır
    parser = argparse.ArgumentParser(description="Geliştirilmiş CBAM entegrasyonlu YOLOv8 modelini eğit")
    parser.add_argument('--epochs', type=int, default=100, help="Eğitim devir sayısı")
    parser.add_argument('--batch', type=int, default=16, help="Batch boyutu")
    parser.add_argument('--patience', type=int, default=20, help="Erken durdurma sabır süresi")
    parser.add_argument('--device', type=str, default='0', help="Kullanılacak cihaz (cpu veya GPU indeksi)")
    
    args = parser.parse_args()
    
    # Modeli eğit
    train_improved_cbam_model(
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        device=args.device
    )

if __name__ == "__main__":
    main()
