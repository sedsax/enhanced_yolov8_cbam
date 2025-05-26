from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def compare_model_performance():
    # Modelleri yükle
    standard_model = YOLO('runs/detect/train5/weights/best.pt')  # Standart model
    cbam_model = YOLO('runs/detect/cbam_final2/weights/best.pt')  # CBAM modeli
    
    # Validasyon datasetinde değerlendir
    standard_results = standard_model.val()
    cbam_results = cbam_model.val()
    
    # Sonuçları yazdır
    print("\n--- Model Performans Karşılaştırması ---")
    print(f"Standart YOLOv8: mAP50={standard_results.box.map50:.4f}, mAP50-95={standard_results.box.map:.4f}")
    print(f"CBAM YOLOv8: mAP50={cbam_results.box.map50:.4f}, mAP50-95={cbam_results.box.map:.4f}")
    print(f"Göreceli İyileştirme: {((cbam_results.box.map - standard_results.box.map) / standard_results.box.map) * 100:.2f}%")
    
    return standard_results, cbam_results

if __name__ == "__main__":
    compare_model_performance()