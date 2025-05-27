"""
YOLOv8 Model Mimarisi Analiz Scripti
Bu script YOLOv8 modelinin mimarisini detaylı olarak analiz eder ve
CBAM modüllerinin nereye yerleştirilebileceğine dair öneriler sunar.
"""

import os
import torch
import torch.nn as nn
import json
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Renk ayarları
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

class YOLOv8Analyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.nn_model = self.model.model
        self.module_tree = {}
        self.module_info = {}
        self.suitable_cbam_locations = []
        
    def analyze_model_structure(self):
        """Model yapısını analiz eder ve detaylı bir rapor oluşturur"""
        print("\n" + "="*80)
        print(f"YOLOv8 MODEL ANALİZİ: {self.model_path}")
        print("="*80)
        
        # Temel model bilgilerini yazdır
        total_params = sum(p.numel() for p in self.nn_model.parameters())
        trainable_params = sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)
        
        print(f"\nToplam parametreler: {total_params:,}")
        print(f"Eğitilebilir parametreler: {trainable_params:,}")
        print(f"Model boyutu: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Ana modülleri listele
        print("\nAna Modüller:")
        for name, module in self.nn_model.named_children():
            print(f"- {name}: {module.__class__.__name__}")
            params = sum(p.numel() for p in module.parameters())
            print(f"  Parametreler: {params:,} ({params/total_params*100:.2f}%)")
            
        # Modül ağacını oluştur
        self._build_module_tree()
        
        # Modülleri katmanlara göre analiz et
        self._analyze_by_type()
        
        # CBAM için uygun konumları belirle
        self._find_suitable_cbam_locations()
        
        # Görselleştirme
        self._visualize_model_complexity()
        
        return self
        
    def _build_module_tree(self):
        """Model modüllerinin hiyerarşik yapısını oluşturur"""
        # Her modül için bilgileri topla
        for name, module in self.nn_model.named_modules():
            module_type = module.__class__.__name__
            num_params = sum(p.numel() for p in module.parameters())
            structure = {}
            
            # Modülün hiyerarşik yolu
            parts = name.split('.')
            path = []
            
            # Modül bilgilerini kaydet
            self.module_info[name] = {
                'type': module_type,
                'params': num_params,
                'path': parts,
                'depth': len(parts)
            }
            
            # Conv2d modülleri için ekstra bilgiler
            if isinstance(module, nn.Conv2d):
                self.module_info[name]['in_channels'] = module.in_channels
                self.module_info[name]['out_channels'] = module.out_channels
                self.module_info[name]['kernel_size'] = module.kernel_size
                
        # Derinlik analizi 
        depths = defaultdict(int)
        for name, info in self.module_info.items():
            depths[info['depth']] += 1
            
        print("\nModel derinlik analizi:")
        for depth, count in sorted(depths.items()):
            print(f"Derinlik {depth}: {count} modül")
    
    def _analyze_by_type(self):
        """Modülleri türlerine göre analiz eder"""
        module_types = defaultdict(list)
        for name, info in self.module_info.items():
            module_types[info['type']].append(name)
            
        print("\nModül türleri analizi:")
        for module_type, modules in sorted(module_types.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"{module_type}: {len(modules)} adet")
            
            # Konvolüsyonel katmanların kanal bilgilerini göster
            if module_type == 'Conv2d' and len(modules) < 20:  # Çok fazla değilse detay göster
                for name in modules[:5]:  # İlk 5 modülü göster
                    info = self.module_info[name]
                    print(f"  - {name}: {info.get('in_channels', '?')} -> {info.get('out_channels', '?')} kanallar")
    
    def _find_suitable_cbam_locations(self):
        """CBAM modülleri için en uygun yerleri belirler"""
        print("\nCBAM için uygun konumlar:")
        
        # Backbone ve head modüllerindeki önemli lokasyonlar
        backbone_modules = []
        head_modules = []
        
        # Konumları belirle
        for name, info in self.module_info.items():
            # C2f bloklarında CBAM kullanımı en etkili olacaktır
            if 'backbone' in name and 'C2f' in name and info['type'] not in ['Conv2d', 'BatchNorm2d', 'SiLU']:
                backbone_modules.append((name, info))
                
            # Detection head içinde de CBAM yararlı olabilir
            elif 'head' in name and info['type'] not in ['Conv2d', 'BatchNorm2d', 'SiLU']:
                head_modules.append((name, info))
        
        # Backbone modülleri için öneriler
        print("\nBackbone'da önerilen CBAM konumları:")
        for name, info in backbone_modules:
            if 'C2f' in name:
                # Kanal sayısını bulalım
                channels = None
                for subname, subinfo in self.module_info.items():
                    if subname.startswith(name) and 'Conv2d' in subinfo['type'] and 'out_channels' in subinfo:
                        channels = subinfo['out_channels']
                        break
                
                if channels:
                    print(f"- {name} (Kanallar: {channels})")
                    self.suitable_cbam_locations.append((name, channels))
        
        # Head modülleri için öneriler
        print("\nDetection head'de önerilen CBAM konumları:")
        for name, info in head_modules:
            if any(x in name for x in ['cv2', 'cv3']):
                # Kanal sayısını bulalım
                channels = None
                for subname, subinfo in self.module_info.items():
                    if subname.startswith(name) and 'Conv2d' in subinfo['type'] and 'out_channels' in subinfo:
                        channels = subinfo['out_channels']
                        break
                        
                if channels:
                    print(f"- {name} (Kanallar: {channels})")
                    self.suitable_cbam_locations.append((name, channels))
        
        # En iyi CBAM yerleşim stratejisini öner
        print("\nÖnerilen CBAM entegrasyon stratejisi:")
        if self.suitable_cbam_locations:
            print(f"Toplam {len(self.suitable_cbam_locations)} konuma CBAM modülü eklenebilir.")
            print("CBAM modülleri aşağıdaki konumlara eklenmelidir:")
            for i, (name, channels) in enumerate(self.suitable_cbam_locations[:5], 1):
                print(f"{i}. {name} ({channels} kanal)")
            
            # CBAM ekleme kodu örneği
            print("\nCBAM ekleme için örnek kod:")
            print("```python")
            print("cbam_locations = [")
            for name, channels in self.suitable_cbam_locations[:5]:
                print(f"    ('{name}', {channels}),")
            print("]")
            print("```")
        else:
            print("Uygun CBAM konumu bulunamadı.")
    
    def _visualize_model_complexity(self):
        """Model karmaşıklığını görselleştirir"""
        # Modül türlerine göre parametre dağılımı
        module_types = defaultdict(int)
        for name, info in self.module_info.items():
            if '.' not in name:  # Sadece üst seviye modüller
                module_types[info['type']] += info['params']
        
        # Sonuçları kaydet
        output_dir = "model_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Görselleştirmeleri oluştur
        plt.figure(figsize=(12, 6))
        plt.bar(module_types.keys(), module_types.values())
        plt.title("YOLOv8 Model Karmaşıklığı - Ana Modüller")
        plt.ylabel("Parametre Sayısı")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_complexity.png"))
        
        # CBAM konumlarını JSON olarak kaydet
        with open(os.path.join(output_dir, "cbam_locations.json"), 'w') as f:
            json.dump(self.suitable_cbam_locations, f, indent=2)
            
        print(f"\nModel analiz sonuçları '{output_dir}' klasörüne kaydedildi.")
        
    def get_cbam_locations(self):
        """CBAM için uygun konumları döndürür"""
        return self.suitable_cbam_locations

def main():
    print("YOLOv8 Model Mimarisi Analizi Başlatılıyor...")
    model_path = "yolov8n.pt"  # Model dosyasının yolu
    
    analyzer = YOLOv8Analyzer(model_path)
    analyzer.analyze_model_structure()
    
    print("\nAnaliz tamamlandı!")
    print("\nBu bilgileri kullanarak CBAM modüllerinin entegrasyonunu iyileştirebilirsiniz.")

if __name__ == "__main__":
    main()
