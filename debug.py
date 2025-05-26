from ultralytics import YOLO
import torch

def inspect_model():
    # YOLOv8 modelini yükle
    model = YOLO('yolov8n.pt')
    
    print(f"Model tipi: {type(model).__name__}")
    print(f"Model özellikleri: {dir(model)}")
    
    # Model.model yapısını incele
    if hasattr(model, 'model'):
        print(f"\nModel.model tipi: {type(model.model).__name__}")
        print(f"Model.model özellikleri: {dir(model.model)}")
        
        # Katmanları incele
        if hasattr(model.model, 'model'):
            layers = model.model.model
            print(f"\nKatman sayısı: {len(layers)}")
            
            for i, layer in enumerate(layers):
                print(f"\nKatman {i}: {type(layer).__name__}")
                print(f"Özellikleri: {dir(layer)}")
                
                # Çıkış kanallarını farklı yöntemlerle bulmaya çalış
                channels = None
                
                if hasattr(layer, 'out_channels'):
                    channels = layer.out_channels
                    print(f"  out_channels: {channels}")
                elif hasattr(layer, 'c'):
                    channels = layer.c
                    print(f"  c: {channels}")
                elif hasattr(layer, 'c2'):
                    channels = layer.c2
                    print(f"  c2: {channels}")
                
                # Bazı yaygın nested yapıları kontrol et
                if hasattr(layer, 'conv'):
                    print(f"  layer.conv tipi: {type(layer.conv).__name__}")
                    if hasattr(layer.conv, 'out_channels'):
                        print(f"  layer.conv.out_channels: {layer.conv.out_channels}")
                
                if hasattr(layer, 'cv1'):
                    print(f"  layer.cv1 tipi: {type(layer.cv1).__name__}")
                    if hasattr(layer.cv1, 'out_channels'):
                        print(f"  layer.cv1.out_channels: {layer.cv1.out_channels}")
                
                if hasattr(layer, 'cv2'):
                    print(f"  layer.cv2 tipi: {type(layer.cv2).__name__}")
                    if hasattr(layer.cv2, 'out_channels'):
                        print(f"  layer.cv2.out_channels: {layer.cv2.out_channels}")
                
                # C2f için özel kontrol
                if type(layer).__name__ == 'C2f' and hasattr(layer, 'm'):
                    print(f"  C2f.m modülleri: {len(layer.m)}")
                    if len(layer.m) > 0:
                        print(f"  Son modül tipi: {type(layer.m[-1]).__name__}")
                        if hasattr(layer.m[-1], 'cv2'):
                            print(f"  Son modülün cv2 out_channels: {layer.m[-1].cv2.out_channels}")

if __name__ == "__main__":
    inspect_model()