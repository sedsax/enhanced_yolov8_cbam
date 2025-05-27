"""
YOLOv8 modeline CBAM entegre etmek için iyileştirilmiş yardımcı sınıf
Bu modül, YOLOv8 modelini CBAM modülleri ile güçlendirir.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from cbam import CBAM
import copy
import time

class ImprovedYOLOv8WithCBAM:
    def __init__(self, model_path, cbam_locations=None, debug=True):
        """
        YOLOv8 modelini CBAM ile güçlendir
        
        Args:
            model_path: YOLOv8 model dosyasının yolu
            cbam_locations: CBAM eklenecek (module_name, channels) listesi
                           None ise otomatik olarak belirlenir
            debug: Detaylı logları gösterme
        """
        self.debug = debug
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.nn_model = self.model.model
        
        # Model parametre bilgileri
        self.original_params = sum(p.numel() for p in self.nn_model.parameters())
        
        if self.debug:
            print(f"\nOrijinal YOLOv8 modeli yüklendi: {model_path}")
            print(f"Parametreler: {self.original_params:,}")
        
        # Model modüllerini analiz et ve CBAM için uygun konumları bul
        if cbam_locations is None:
            cbam_locations = self._find_optimal_cbam_locations()
        
        # CBAM modüllerini yerleştir
        self._add_cbam_modules(cbam_locations)
        
        # Güncellenmiş model parametreleri
        self.enhanced_params = sum(p.numel() for p in self.nn_model.parameters())
        
        if self.debug:
            print(f"\nCBAM entegrasyonu tamamlandı")
            print(f"Orijinal parametreler: {self.original_params:,}")
            print(f"Güncellenmiş parametreler: {self.enhanced_params:,}")
            print(f"Ek parametreler: {self.enhanced_params - self.original_params:,}")
    
    def _find_optimal_cbam_locations(self):
        """YOLOv8 modelinde CBAM için en uygun konumları belirle"""
        if self.debug:
            print("\nOptimum CBAM konumları belirleniyor...")
        
        cbam_locations = []
        backbone_targets = []
        head_targets = []
        
        # İlk geçiş: Model mimarisini incele ve anahtar bileşenleri belirle
        for name, module in self.nn_model.named_modules():
            # Backbone C2f blokları - YOLOv8'in ana özellik çıkarıcıları
            if "backbone.C2f" in name and len(name.split('.')) <= 3:
                backbone_targets.append(name)
                
            # Head bileşenleri - Tespit/sınıflandırma için özellik işleme
            elif "head." in name and any(x in name for x in ["cv2", "cv3"]) and len(name.split('.')) <= 3:
                head_targets.append(name)
        
        # İşlenecek modülleri seçme
        target_modules = backbone_targets + head_targets
        
        # İkinci geçiş: Modüllerin kanal sayılarını belirle
        if self.debug:
            print(f"Toplam {len(target_modules)} potansiyel CBAM konumu bulundu")
        
        for module_name in target_modules:
            channel_count = 0
            module = None
            
            # Modülü bul
            try:
                parts = module_name.split('.')
                current = self.nn_model
                for part in parts:
                    current = getattr(current, part) if not part.isdigit() else current[int(part)]
                module = current
            except (AttributeError, IndexError) as e:
                if self.debug:
                    print(f"  - {module_name} modülü bulunamadı: {e}")
                continue
                
            # Kanal sayısını bul
            if hasattr(module, "cv1") and hasattr(module.cv1, "conv") and hasattr(module.cv1.conv, "out_channels"):
                channel_count = module.cv1.conv.out_channels
            elif hasattr(module, "conv") and hasattr(module.conv, "out_channels"):
                channel_count = module.conv.out_channels
            else:
                # Son çare: konvolüsyon katmanlarını ara
                for name, submodule in module.named_modules():
                    if isinstance(submodule, nn.Conv2d) and submodule.out_channels >= 64:
                        channel_count = submodule.out_channels
                        break
            
            if channel_count > 0:
                cbam_locations.append((module_name, channel_count))
                if self.debug:
                    print(f"  + {module_name} konumuna CBAM ekleniyor (Kanallar: {channel_count})")
        
        if self.debug:
            print(f"\nToplam {len(cbam_locations)} CBAM konumu belirlendi.")
            
        return cbam_locations
    
    def _add_cbam_modules(self, locations):
        """
        Belirlenen konumlara CBAM modüllerini entegre et
        
        Args:
            locations: (module_name, channels) tuple'larının listesi
        """
        if self.debug:
            print("\nCBAM MODÜLLERİ EKLENİYOR")
            print("-" * 50)
        
        total_cbam_modules = 0
        start_time = time.time()
        
        for module_name, channels in locations:
            # Modülü bul
            try:
                # Modül yolunu noktalar üzerinden ayır
                parts = module_name.split('.')
                module_parent = self.nn_model
                
                # Son kısım hariç tüm parçalara git (ebeveyn modülü bul)
                for i, part in enumerate(parts[:-1]):
                    module_parent = getattr(module_parent, part) if not part.isdigit() else module_parent[int(part)]
                
                # Son parça - hedef modülün adı
                last_part = parts[-1]
                target_module = getattr(module_parent, last_part) if not last_part.isdigit() else module_parent[int(last_part)]
                
                if self.debug:
                    print(f"\nHedef modül bulundu: {module_name}")
                    print(f"  Tür: {type(target_module).__name__}")
                    print(f"  Parametreler: {sum(p.numel() for p in target_module.parameters()):,}")
                
                # İlk deneme: Modülü Sequential ile sarmala
                new_module = nn.Sequential(
                    copy.deepcopy(target_module),
                    CBAM(channels)
                )
                
                # Modülü değiştir
                if last_part.isdigit():
                    module_parent[int(last_part)] = new_module
                else:
                    setattr(module_parent, last_part, new_module)
                
                # Başarı mesajı
                if self.debug:
                    print(f"  ✓ {module_name} konumuna CBAM başarıyla eklendi")
                    print(f"    Yeni modül parametreleri: {sum(p.numel() for p in new_module.parameters()):,}")
                    print(f"    CBAM parametreleri: {sum(p.numel() for p in new_module[1].parameters()):,}")
                
                total_cbam_modules += 1
                
            except (AttributeError, IndexError, ValueError) as e:
                if self.debug:
                    print(f"  ✗ {module_name} konumuna CBAM eklenemedi: {e}")
        
        # Sonuçları göster
        elapsed_time = time.time() - start_time
        if self.debug:
            print(f"\nToplam {total_cbam_modules} CBAM modülü eklendi ({elapsed_time:.2f} saniyede)")
            print("-" * 50)
            
            # Model yapısını kontrol et
            print("\nCBAM entegrasyonu sonrası model yapısı doğrulanıyor...")
            modified_params = sum(p.numel() for p in self.nn_model.parameters())
            added_params = modified_params - self.original_params
            if added_params > 0:
                print(f"✓ Model parametreleri artış gösterdi: +{added_params:,}")
                print(f"  CBAM entegrasyonu başarılı görünüyor!")
            else:
                print(f"✗ Model parametrelerinde değişiklik yok. CBAM entegrasyonu başarısız olmuş olabilir.")
            print("-" * 50)
    
    def train(self, **kwargs):
        """Modeli eğit (orijinal YOLO API'si ile)"""
        return self.model.train(**kwargs)
    
    def val(self, **kwargs):
        """Modeli doğrula (orijinal YOLO API'si ile)"""
        return self.model.val(**kwargs)
    
    def predict(self, **kwargs):
        """Tahmin yap (orijinal YOLO API'si ile)"""
        return self.model.predict(**kwargs)
    
    def export(self, **kwargs):
        """Modeli dışa aktar (orijinal YOLO API'si ile)"""
        return self.model.export(**kwargs)
    
    def info(self, **kwargs):
        """Model bilgilerini getir (orijinal YOLO API'si ile)"""
        return self.model.info(**kwargs)
    
    def get_module_names(self):
        """Modeldeki tüm modül adlarını listeler"""
        return [name for name, _ in self.nn_model.named_modules()]
