from models.custom_yolov8 import CustomYOLOv8
import torch

def test_model_structure():
    try:
        # Özel modeli yükle
        model = CustomYOLOv8('yolov8n.pt')
        print("Model başarıyla oluşturuldu.")
        
        # Test girdisi oluştur
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # İleri geçiş testi
        outputs = model.model(dummy_input)
        print(f"Model çıktısı başarıyla oluşturuldu.")
        
        # Eklenen CBAM modüllerini kontrol et
        cbam_found = False
        for name, module in model.model.named_modules():
            if 'cbam' in str(type(module)).lower():
                cbam_found = True
                print(f"CBAM modülü bulundu: {name}")
        
        if not cbam_found:
            print("UYARI: CBAM modülü bulunamadı!")
        
        # Ek tespit katmanını kontrol et
        if hasattr(model.model.model, 'extra_layer'):
            print("Ek tespit katmanı bulundu.")
        else:
            print("UYARI: Ek tespit katmanı bulunamadı!")
            
        return True
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_structure()