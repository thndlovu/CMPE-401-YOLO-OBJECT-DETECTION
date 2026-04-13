from ultralytics import YOLO

DATA_YAML = 'data/visdrone/yolo/data.yaml'
PROJECT = 'experiments/results'

experiments = [
    {
        'name': 'exp1_yolo11n',
        'model': 'yolo11n.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
    },
    {
        'name': 'exp2_yolo11s_res832',
        'model': 'yolo11s.pt',
        'epochs': 50,
        'imgsz': 832,
        'batch': 8,  # smaller batch due to higher resolution
    },
    {
        'name': 'exp3_yolo11m',
        'model': 'yolo11m.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
    },
    {
        'name': 'exp4_improvement_yolo11m_res832',
        'model': 'yolo11m.pt',
        'epochs': 100,
        'imgsz': 832,
        'batch': 8,
    },
]

def run_experiment(exp):
    print(f"\n{'='*50}")
    print(f"Running: {exp['name']}")
    print(f"{'='*50}")
    
    model = YOLO(exp['model'])
    
    results = model.train(
        data=DATA_YAML,
        epochs=exp['epochs'],
        imgsz=exp['imgsz'],
        batch=exp['batch'],
        project=PROJECT,
        name=exp['name'],
        device=0,
        patience=15,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )
    
    print(f"\nCompleted: {exp['name']}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=None,
                        help='Experiment index to run (0-3). If not set, runs all.')
    args = parser.parse_args()
    
    if args.exp is not None:
        run_experiment(experiments[args.exp])
    else:
        for exp in experiments:
            run_experiment(exp)