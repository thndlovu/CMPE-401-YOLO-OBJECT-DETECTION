from ultralytics import YOLO

DATA_YAML = 'data/visdrone/yolo/data.yaml'
PROJECT = 'experiments/results'

comparison_models = [
    {
        'name': 'compare_yolov8s',
        'model': 'yolov8s.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
    },
    {
        'name': 'compare_yolov9s',
        'model': 'yolov9s.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
    },
    {
        'name': 'compare_yolov10s',
        'model': 'yolov10s.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
    },
    {
        'name': 'compare_yolov5su',
        'model': 'yolov5su.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
    },
]

def run_comparison(model_config):
    print(f"\n{'='*50}")
    print(f"Running: {model_config['name']}")
    print(f"{'='*50}")
    
    model = YOLO(model_config['model'])
    
    results = model.train(
        data=DATA_YAML,
        epochs=model_config['epochs'],
        imgsz=model_config['imgsz'],
        batch=model_config['batch'],
        project=PROJECT,
        name=model_config['name'],
        device=0,
        patience=15,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )
    
    print(f"\nCompleted: {model_config['name']}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=None,
                        help='Model index to run (0-3). If not set, runs all.')
    args = parser.parse_args()
    
    if args.model is not None:
        run_comparison(comparison_models[args.model])
    else:
        for m in comparison_models:
            run_comparison(m)