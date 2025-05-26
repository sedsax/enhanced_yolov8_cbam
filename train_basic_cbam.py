from models.yolov8_cbam import YOLOv8_CBAM
import torch
import cv2
import numpy as np

def main():
    print("CBAM ile geliştirilmiş YOLOv8 modeli oluşturuluyor...")
    model = YOLOv8_CBAM('yolov8n.pt')
    
    # Model yapısını kontrol et
    if hasattr(model.model, 'model'):
        print("\nModel katmanları:")
        for i, layer in enumerate(model.model.model):
            print(f"Katman {i}: {type(layer).__name__}")
            if 'CBAM_C2f' in type(layer).__name__:
                print(f"  -> CBAM başarıyla entegre edildi!")
    
    # Daha basit bir test: bir tensör ile forward geçişi yap
    print("\nBasit tensör testi yapılıyor...")
    try:
        # Test için dummy tensor oluştur
        x = torch.randn(1, 3, 640, 640)
        # Gerçek forward geçişi yerine model içindeki işlemlerle test et
        # İlk katmanı çağır
        x = model.model.model[0](x)
        print(f"Katman 0 çıkış boyutu: {x.shape}")
        
        # İkinci katmanı çağır
        x = model.model.model[1](x)
        print(f"Katman 1 çıkış boyutu: {x.shape}")
        
        # CBAM eklediğimiz 4. katmanı test et
        x = model.model.model[2](x)
        x = model.model.model[3](x)
        print(f"CBAM öncesi giriş boyutu: {x.shape}")
        x = model.model.model[4](x)
        print(f"CBAM sonrası çıkış boyutu: {x.shape}")
        
        print("Tensör testi başarılı!")
        return True
    except Exception as e:
        print(f"Tensör testi sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()