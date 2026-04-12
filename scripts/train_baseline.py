from ultralytics import YOLO
import yaml
import os

# Config
DATA_YAML = 'data/visdrone/yolo/data.yaml'
MODEL = 'yolo11s.pt'  # small model for baseline
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT = 'experiments/results'
NAME = 'baseline_yolo11s'

def main():
    # Load model
    model = YOLO(MODEL)
    
    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        device=0,  # GPU
        patience=10,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )
    
    print("\n=== Baseline Training Complete ===")
    print(f"Results saved to: {PROJECT}/{NAME}")

if __name__ == '__main__':
    main()