from ultralytics import YOLO
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.cbam import CBAM

class CBAM_C2f(nn.Module):
    """CBAM ile geliştirilmiş C2f modülü"""
    def __init__(self, original_c2f):
        super().__init__()
        self._c2f = original_c2f
        
        # Doğru çıkış kanalı sayısını belirle
        if hasattr(original_c2f, 'c2'):
            self.c = original_c2f.c2
        elif hasattr(original_c2f, 'c'):
            self.c = original_c2f.c * 2
        else:
            try:
                dummy_input = torch.zeros(1, 64, 16, 16)
                with torch.no_grad():
                    dummy_output = original_c2f(dummy_input)
                self.c = dummy_output.shape[1]
            except Exception as e:
                self.c = 64
                print(f"Kanal sayısı belirlenemedi, varsayılan 64 kullanılıyor")
                
        print(f"CBAM için {self.c} kanal sayısı kullanılıyor")
        self.cbam = CBAM(self.c)
        
        # Orijinal modülün kritik özelliklerini kopyala
        self.f = getattr(original_c2f, 'f', -1)
        self.i = getattr(original_c2f, 'i', 0)
        self.type = getattr(original_c2f, 'type', 'c2f')
        
        # Diğer önemli öznitelikleri kopyala
        for attr in ['cv1', 'cv2', 'cv3', 'cv4', 'bn', 'act', 'm']:
            if hasattr(original_c2f, attr):
                setattr(self, attr, getattr(original_c2f, attr))
    
    def forward(self, x):
        x = self._c2f(x)
        if hasattr(self, '_debug') and self._debug:
            print(f"C2f çıkış boyutu: {x.shape}")
            print(f"CBAM yapılandırma kanalları: {self.c}")
        return self.cbam(x)

class YOLOv8_CBAM(YOLO):
    def __init__(self, model_path='yolov8n.pt'):
        super().__init__(model_path)
        
        print("YOLOv8 yapısı CBAM ile değiştiriliyor...")
        
        if hasattr(self.model, 'model'):
            # Daha fazla C2f katmanına CBAM ekleyin
            backbone_indices = [4, 6, 8, 12, 15, 18, 21]  # Daha fazla katman eklendi
            
            for i in backbone_indices:
                if i < len(self.model.model):
                    layer = self.model.model[i]
                    print(f"Katman {i} ({type(layer).__name__}) inceleniyor")
                    if 'C2f' in type(layer).__name__:
                        print(f"Katman {i}'e CBAM ekleniyor")
                        wrapper = CBAM_C2f(layer)
                        wrapper._debug = True
                        self.model.model[i] = wrapper
                        print(f"Katman {i} için CBAM başarıyla eklendi")
        
        print("Model başarıyla değiştirildi")