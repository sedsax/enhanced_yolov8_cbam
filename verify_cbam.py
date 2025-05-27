import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from yolov8_cbam import YOLOv8WithCBAM
import cv2

# Set up paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml')
MODEL_PATH = os.path.join(ROOT_DIR, 'yolov8n.pt')
CBAM_MODEL_PATH = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_cbam', 'weights', 'best.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'cbam_visualization')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sample_image():
    """Get a sample image from the dataset"""
    test_img_dir = os.path.join(ROOT_DIR, 'dataset', 'test', 'images')
    img_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    if not img_files:
        raise ValueError("No image files found in test directory")
    
    img_path = os.path.join(test_img_dir, img_files[0])
    print(f"Using sample image: {img_path}")
    return img_path

def create_fresh_cbam_model(verbose=True):
    """Create a fresh YOLOv8 model with CBAM modules integrated"""
    print("\nCreating a new YOLOv8 model with CBAM integration...")
    model = YOLOv8WithCBAM(MODEL_PATH)
    
    if verbose:
        # Print model summary
        print("\nModel Summary:")
        print(f"Original YOLOv8 parameters: 3,005,843")  # Known value for YOLOv8n
        total_params = sum(p.numel() for p in model.nn_model.parameters())
        print(f"CBAM-enhanced parameters: {total_params}")
        print(f"Additional parameters: {total_params - 3005843}")
    
    return model

def visualize_models(sample_img_path):
    """Compare predictions of the standard and CBAM-enhanced models"""
    # Load the trained models
    standard_model = YOLO(os.path.join(ROOT_DIR, 'runs', 'detect', 'train_basic', 'weights', 'best.pt'))
    cbam_model = YOLO(CBAM_MODEL_PATH)
    
    # Run prediction
    standard_results = standard_model.predict(sample_img_path, conf=0.25)
    cbam_results = cbam_model.predict(sample_img_path, conf=0.25)
    
    # Load the original image
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Plot results
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.title("Standard YOLOv8")
    plt.imshow(img)
    for box in standard_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2))
    
    plt.subplot(1, 2, 2)
    plt.title("YOLOv8 with CBAM")
    plt.imshow(img)
    for box in cbam_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='blue', linewidth=2))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(output_path)
    print(f"\nSaved model comparison visualization to {output_path}")

def main():
    print("Starting CBAM verification process...")
    
    try:
        # Get a sample image
        sample_img_path = get_sample_image()
        
        # Create and examine a fresh CBAM model
        create_fresh_cbam_model()
        
        # Check if the trained models exist
        standard_model_path = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_basic', 'weights', 'best.pt')
        cbam_model_path = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_cbam', 'weights', 'best.pt')
        
        print(f"\nChecking for trained models:")
        print(f"Standard model: {'EXISTS' if os.path.exists(standard_model_path) else 'NOT FOUND'}")
        print(f"CBAM model: {'EXISTS' if os.path.exists(cbam_model_path) else 'NOT FOUND'}")
        
        if not os.path.exists(standard_model_path) or not os.path.exists(cbam_model_path):
            print("\nWARNING: One or both trained models not found.")
            print("Will skip visualization that requires trained models.")
        else:
            # Visualize and compare models
            print("\nVisualizing trained models...")
            visualize_models(sample_img_path)
        
        print("\nYOLOv8 vs YOLOv8-CBAM Verification Complete!")
        print("Check the visualizations in the 'cbam_visualization' directory.")
    except Exception as e:
        import traceback
        print(f"\nERROR during verification: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
