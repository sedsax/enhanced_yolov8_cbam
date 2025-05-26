from ultralytics import YOLO
import torch.nn as nn
import torch
import types
import inspect

from models.cbam import CBAM
from models.additional_layer import AdditionalDetectionLayer

class CustomYOLOv8(YOLO):
    def __init__(self, model='yolov8n.pt', task=None):
        # Önce orijinal YOLO modelini yükle
        super().__init__(model, task)
        
        # Modeli düzenle
        self._modify_model()
    
    def _get_channel_info(self, layer):
        """YOLOv8 katmanlarından kanal bilgisini çıkarma"""
        
        # Direkt olarak kanal bilgisine sahip olabilecek öznitelikler
        if hasattr(layer, 'c'):
            return layer.c
        
        # Düzgün Conv2d yapısına sahip olan iç nesneye erişim
        if hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
            return layer.conv.out_channels
            
        # YOLOv8 özel Conv sınıfı için
        if type(layer).__name__ == 'Conv':
            if hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
                return layer.conv.out_channels
        
        # C2f için özel durum
        if type(layer).__name__ == 'C2f':
            if hasattr(layer, 'c'):
                return layer.c
            elif hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv') and hasattr(layer.cv1.conv, 'out_channels'):
                return layer.cv1.conv.out_channels * 2  # C2f için genelde çıkış 2 katıdır
        
        # Detect katmanı için
        if 'Detect' in type(layer).__name__:
            if hasattr(layer, 'nc'):
                return layer.nc
        
        # Katmanın tüm özniteliklerini dolaş ve muhtemel kanal bilgisini bul
        for name, value in vars(layer).items():
            if name in ['c', 'channels', 'out_channels', 'ch']:
                return value
            
            # İç içe yapıları kontrol et
            elif hasattr(value, 'out_channels'):
                return value.out_channels
        
        # Hiçbir bilgi bulunamadıysa None döndür
        return None
    
    def _modify_model(self):
        # YOLOv8 model yapısını incele
        print("Model yapısı inceleniyor...")
        
        # Model yapısını kontrol et
        if hasattr(self.model, 'model'):
            model_layers = self.model.model
            print("Model katmanlarına erişildi.")
        else:
            print("UYARI: Model katmanlarına erişilemedi!")
            return
        
        # Model yapısı hakkında bilgi ver
        print(f"\nToplam katman sayısı: {len(model_layers)}")
        
        # Backbone katmanlarına CBAM ekle
        print("\nBackbone'a CBAM modülleri ekleniyor...")
        
        # YOLOv8n için tipik backbone katmanları
        # Farklı model boyutları için bu indeksler değişebilir
        backbone_indices = [4, 6, 10]  # YOLOv8n'de CBAM ekleyeceğimiz katmanlar
        
        for i in backbone_indices:
            if i < len(model_layers):
                layer = model_layers[i]
                print(f"Katman {i} ({type(layer).__name__}) inceleniyor...")
                
                # Katman çıkış kanallarını belirle
                channels = self._get_channel_info(layer)
                
                if channels:
                    print(f"Katman {i} için {channels} kanallı CBAM ekleniyor")
                    cbam_module = CBAM(channels)
                    
                    # Mevcut forward fonksiyonunu kaydet
                    original_forward = layer.forward
                    
                    # Yeni forward fonksiyonu tanımla
                    def new_forward(self_inner, x, original_function=original_forward, cbam=cbam_module):
                        x = original_function(x)
                        return cbam(x)
                    
                    # Yeni forward fonksiyonunu atama
                    layer.forward = types.MethodType(new_forward, layer)
                    print(f"Katman {i} için CBAM başarıyla eklendi")
                else:
                    print(f"Katman {i} için kanal sayısı belirlenemedi, CBAM eklenemedi")
        
        # Ek tespit katmanı ekle
        print("\nEk tespit katmanı ekleniyor...")
        
        # Tespit katmanını bul
        detect_layer = None
        detect_index = -1
        
        for i, layer in enumerate(model_layers):
            if 'Detect' in type(layer).__name__:
                detect_layer = layer
                detect_index = i
                print(f"Tespit katmanı bulundu: Index {i}, Tip {type(layer).__name__}")
                break
        
        if detect_layer:
            # Tespit katmanı sınıf sayısını al
            num_classes = detect_layer.nc if hasattr(detect_layer, 'nc') else 80
            print(f"Tespit katmanında {num_classes} sınıf bulundu")
            
            # Son FPN katmanından kanal sayısı al
            if detect_index > 0:
                prev_layer = model_layers[detect_index - 1]
                in_channels = self._get_channel_info(prev_layer)
                
                if in_channels:
                    print(f"Önceki katmandan {in_channels} giriş kanalı alındı")
                else:
                    # Kanal sayısını bulamadıysak varsayılan değer kullan
                    in_channels = 256
                    print("Kanal sayısı belirlenemedi, varsayılan 256 kullanılıyor")
                
                # Ek tespit katmanı oluştur
                self.additional_layer = AdditionalDetectionLayer(
                    in_channels=in_channels,
                    out_channels=256,
                    num_classes=num_classes
                )
                
                print("Ek tespit katmanı eklendi")
                
                # Modeli başarıyla değiştirdiğimizi belirt
                setattr(self, 'modified', True)
            else:
                print("UYARI: Tespit katmanından önceki katman bulunamadı")
        else:
            print("UYARI: Tespit katmanı bulunamadı!")
            
        print("Model modifikasyonu tamamlandı!")

# custom_yolov8.py dosyasına eklenecek kod
def forward_with_debug(self_inner, x, original_function, cbam):
    print(f"CBAM çağrıldı: {type(cbam).__name__}")
    x_orig = original_function(x)
    x_cbam = cbam(x_orig)
    print(f"CBAM öncesi ve sonrası norm değişimi: {torch.norm(x_orig).item():.4f} -> {torch.norm(x_cbam).item():.4f}")
    return x_cbam 