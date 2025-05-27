import os
from yolov8_cbam import YOLOv8WithCBAM

# Set up paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml')
MODEL_PATH = os.path.join(ROOT_DIR, 'yolov8n.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'runs', 'train_cbam')

def train_cbam_model():
    print("Training YOLOv8n model with CBAM modules...")
    
    # Load the model with CBAM attention modules
    model = YOLOv8WithCBAM(MODEL_PATH)
      # Train the model
    model.train(
        data=DATA_YAML,          # Path to data config file
        epochs=30,              # Increased number of epochs to improve learning
        imgsz=640,               # Image size
        batch=16,                # Batch size
        name='train_cbam',       # Run name
        device='0',              # Device to use (CPU=cpu, GPU=0,1,2...)
        patience=10,             # Early stopping patience
        lr0=0.001,               # Initial learning rate
        lrf=0.01,                # Final learning rate (fraction of lr0)
        warmup_epochs=3,         # Warmup epochs
        weight_decay=0.0005      # Weight decay for optimizer
    )
    
    print(f"Training completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_cbam_model()
