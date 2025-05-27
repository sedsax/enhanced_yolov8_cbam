import os
import argparse
import subprocess
import time

def run_command(command):
    """Run a command and print its output in real-time"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    process.stdout.close()
    return_code = process.wait()
    return return_code

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 training and comparison')
    parser.add_argument('--skip-basic', action='store_true', help='Skip training the basic YOLOv8 model')
    parser.add_argument('--skip-cbam', action='store_true', help='Skip training the CBAM-enhanced YOLOv8 model')
    parser.add_argument('--compare-only', action='store_true', help='Only run the comparison')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Train basic YOLOv8 model
    if not args.skip_basic and not args.compare_only:
        print("\n===== TRAINING BASIC YOLOv8 MODEL =====")
        run_command('python train_basic.py')
    
    # Train CBAM-enhanced YOLOv8 model
    if not args.skip_cbam and not args.compare_only:
        print("\n===== TRAINING YOLOv8 MODEL WITH CBAM =====")
        run_command('python train_cbam.py')
    
    # Compare the two models
    print("\n===== COMPARING MODELS =====")
    run_command('python compare_models.py')
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()
