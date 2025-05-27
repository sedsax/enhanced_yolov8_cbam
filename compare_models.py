import os
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from yolov8_cbam import YOLOv8WithCBAM

# Set up paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml')
BASIC_MODEL_PATH = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_basic', 'weights', 'best.pt')
CBAM_MODEL_PATH = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_cbam', 'weights', 'best.pt')
RESULTS_DIR = os.path.join(ROOT_DIR, 'comparison_results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_models():
    """Load both trained models"""
    basic_model = None
    cbam_model = None
    
    # Check if models exist before loading
    if os.path.exists(BASIC_MODEL_PATH):
        try:
            # Use offline mode to avoid downloading from GitHub
            basic_model = YOLO(BASIC_MODEL_PATH, task='detect')
            print(f"Successfully loaded basic model from {BASIC_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading basic model: {e}")
    else:
        print(f"Basic model file not found at {BASIC_MODEL_PATH}")
    
    if os.path.exists(CBAM_MODEL_PATH):
        try:
            # Use offline mode to avoid downloading from GitHub
            cbam_model = YOLO(CBAM_MODEL_PATH, task='detect')
            print(f"Successfully loaded CBAM model from {CBAM_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading CBAM model: {e}")
    else:
        print(f"CBAM model file not found at {CBAM_MODEL_PATH}")
    
    return basic_model, cbam_model

def validate_models(basic_model, cbam_model):
    """Run validation on both models and return metrics"""
    basic_metrics = None
    cbam_metrics = None
    
    if basic_model:
        print("\nValidating basic YOLOv8 model...")
        basic_metrics = basic_model.val(data=DATA_YAML, verbose=True)
        
    if cbam_model:
        print("\nValidating CBAM-enhanced YOLOv8 model...")
        cbam_metrics = cbam_model.val(data=DATA_YAML, verbose=True)
    
    return basic_metrics, cbam_metrics

def compare_metrics(basic_metrics, cbam_metrics):
    """Compare and visualize the metrics between the two models"""
    # Extract metrics to compare
    metrics_to_compare = ['precision', 'recall', 'mAP50', 'mAP50-95']
    basic_values = []
    cbam_values = []
    
    if basic_metrics and hasattr(basic_metrics, 'box'):
        basic_values = [
            float(basic_metrics.box.p),  # precision
            float(basic_metrics.box.r),  # recall
            float(basic_metrics.box.map50),  # mAP50
            float(basic_metrics.box.map),  # mAP50-95
        ]
    
    if cbam_metrics and hasattr(cbam_metrics, 'box'):
        cbam_values = [
            float(cbam_metrics.box.p),  # precision
            cbam_metrics.box.r.item() if hasattr(cbam_metrics.box.r, 'item') else float(cbam_metrics.box.r),  # recall
            cbam_metrics.box.map50.item() if hasattr(cbam_metrics.box.map50, 'item') else float(cbam_metrics.box.map50),  # mAP50
            cbam_metrics.box.map.item() if hasattr(cbam_metrics.box.map, 'item') else float(cbam_metrics.box.map),  # mAP50-95
        ]
    
    # Create comparison data with JSON-serializable values
    comparison = {
        'basic_model': {metric: float(value) for metric, value in zip(metrics_to_compare, basic_values)} if basic_values else None,
        'cbam_model': {metric: float(value) for metric, value in zip(metrics_to_compare, cbam_values)} if cbam_values else None
    }
    
    # Save comparison to JSON
    with open(os.path.join(RESULTS_DIR, 'metrics_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Create visualization
    create_comparison_plots(comparison, metrics_to_compare)
    
    return comparison

def create_comparison_plots(comparison, metrics):
    """Create bar plots comparing the metrics"""
    if not comparison['basic_model'] or not comparison['cbam_model']:
        print("Cannot create comparison plots - one or both models missing metrics")
        return
    
    # Prepare data for plotting
    x = range(len(metrics))
    basic_values = [comparison['basic_model'][m] for m in metrics]
    cbam_values = [comparison['cbam_model'][m] for m in metrics]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    plt.bar([i - bar_width/2 for i in x], basic_values, width=bar_width, label='Basic YOLOv8', color='blue', alpha=0.7)
    plt.bar([i + bar_width/2 for i in x], cbam_values, width=bar_width, label='YOLOv8 with CBAM', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Performance Comparison: Basic YOLOv8 vs YOLOv8 with CBAM')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations on bars
    for i, v in enumerate(basic_values):
        plt.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(cbam_values):
        plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'))
    print(f"Comparison plot saved to {os.path.join(RESULTS_DIR, 'metrics_comparison.png')}")
    
def main():
    print("Loading trained models...")
    basic_model, cbam_model = load_models()
    
    if not basic_model and not cbam_model:
        print("Error: Could not load any models. Make sure both models have been trained.")
        return
    
    # Create a partial comparison if only one model is available
    if not basic_model:
        print("Warning: Basic model not available. Will only evaluate CBAM model.")
    
    if not cbam_model:
        print("Warning: CBAM model not available. Will only evaluate Basic model.")
    
    if basic_model or cbam_model:
        print("Running validation on available models...")
        basic_metrics, cbam_metrics = validate_models(basic_model, cbam_model)
        
        print("Comparing model metrics...")
        comparison = compare_metrics(basic_metrics, cbam_metrics)
    else:
        print("No models available for comparison.")
    
    # Print summary
    print("\n===== COMPARISON SUMMARY =====")
    if comparison['basic_model'] and comparison['cbam_model']:
        print("Basic YOLOv8:")
        for metric, value in comparison['basic_model'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nYOLOv8 with CBAM:")
        for metric, value in comparison['cbam_model'].items():
            print(f"  {metric}: {value:.4f}")
            
        # Calculate and display improvement
        print("\nImprovement (CBAM vs Basic):")
        for metric in comparison['basic_model'].keys():
            basic_val = comparison['basic_model'][metric]
            cbam_val = comparison['cbam_model'][metric]
            diff = cbam_val - basic_val
            percent = (diff / basic_val) * 100 if basic_val else 0
            print(f"  {metric}: {diff:.4f} ({percent:+.2f}%)")
    else:
        print("Could not generate complete comparison. Check if both models have been trained.")

if __name__ == "__main__":
    main()
