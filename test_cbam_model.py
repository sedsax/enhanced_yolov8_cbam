from models.yolov8_cbam import YOLOv8_CBAM
import torch
import cv2
import numpy as np
import os

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
    
    # Test tahmini yapma
    print("\nTest görüntüsü oluşturuluyor...")
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    cv2.rectangle(test_img, (100, 100), (300, 300), (255, 0, 0), -1)
    test_img_path = "test_image.jpg"
    cv2.imwrite(test_img_path, test_img)
    
    try:
        print("Test tahmini yapılıyor...")
        results = model.predict(test_img_path, verbose=False)
        print(f"Tahmin tamamlandı: {len(results)} sonuç")
        
        # Sonuçları görselleştir
        for r in results:
            if r.boxes is not None:
                print(f"Tespit edilen nesneler: {len(r.boxes)}")
        
        # Modeli kaydet
        print("\nModel kaydediliyor...")
        save_path = os.path.join(os.getcwd(), 'yolov8n_cbam.pt')
        model.save(save_path)
        print(f"Model kaydedildi: {save_path}")
        
        return True
    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()