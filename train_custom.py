from models.yolov8_cbam import YOLOv8_CBAM
import os

def main():
    try:
        # CBAM ile geliştirilmiş YOLOv8 modeli
        print("CBAM modeli oluşturuluyor...")
        model = YOLOv8_CBAM('yolov8n_cbam.pt')
        print("Model başarıyla oluşturuldu!")
        
        # Eğitim parametrelerini ayarla
        print("Eğitim başlatılıyor...")
        model.train(
            data='data.yaml', 
            epochs=15,         
            imgsz=640,         
            batch=16,          
            device=0,          
            name='cbam_final',   
            verbose=True
        )
        print("Eğitim tamamlandı!")
        
    except Exception as e:
        print(f"Eğitim sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()