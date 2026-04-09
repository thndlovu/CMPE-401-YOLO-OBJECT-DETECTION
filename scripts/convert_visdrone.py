import os
import shutil
from pathlib import Path
from tqdm import tqdm

VISDRONE_CLASSES = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

def convert_annotation(ann_path, img_width, img_height):
    """Convert a single VisDrone annotation file to YOLO format."""
    yolo_lines = []
    
    with open(ann_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 8:
                continue
            
            x_min = int(parts[0])
            y_min = int(parts[1])
            width = int(parts[2])
            height = int(parts[3])
            score = int(parts[4])   # 0 = ignored
            category = int(parts[5])
            
            # Skip ignored regions and unmapped classes
            if score == 0 or category not in VISDRONE_CLASSES:
                continue
            
            # Skip zero-size boxes
            if width == 0 or height == 0:
                continue
            
            # Convert to YOLO format (normalized center x, center y, w, h)
            x_center = (x_min + width / 2) / img_width
            y_center = (y_min + height / 2) / img_height
            norm_w = width / img_width
            norm_h = height / img_height
            
            # Clamp to [0, 1] to handle edge cases
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            class_id = VISDRONE_CLASSES[category]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )
    
    return yolo_lines


def get_image_size(img_path):
    """Get image dimensions without loading full image."""
    from PIL import Image
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def convert_split(visdrone_root, output_root, split):
    """
    Convert one split (train/val/test) of VisDrone to YOLO format.
    
    Expected VisDrone structure:
        visdrone_root/
            VisDrone2019-DET-train/
                images/
                annotations/
            VisDrone2019-DET-val/
            VisDrone2019-DET-test-dev/
    """
    split_map = {
        'train': 'VisDrone2019-DET-train',
        'val':   'VisDrone2019-DET-val',
        'test':  'VisDrone2019-DET-test-dev',
    }
    
    split_dir = Path(visdrone_root) / split_map[split]
    img_dir = split_dir / 'images'
    ann_dir = split_dir / 'annotations'
    
    out_img_dir = Path(output_root) / 'images' / split
    out_lbl_dir = Path(output_root) / 'labels' / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    print(f"\nConverting {split}: {len(images)} images")
    
    skipped = 0
    for img_path in tqdm(images):
        ann_path = ann_dir / (img_path.stem + '.txt')
        
        if not ann_path.exists():
            skipped += 1
            continue
        
        try:
            w, h = get_image_size(img_path)
        except Exception:
            skipped += 1
            continue
        
        yolo_lines = convert_annotation(ann_path, w, h)
        
        # Copy image
        shutil.copy2(img_path, out_img_dir / img_path.name)
        
        # Write label file (even if empty — YOLO expects it)
        out_lbl = out_lbl_dir / (img_path.stem + '.txt')
        with open(out_lbl, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    print(f"  Done. Skipped: {skipped}")


def create_data_yaml(output_root):
    """Create the data.yaml Ultralytics needs."""
    yaml_content = f"""# VisDrone Dataset - YOLO Format
path: {output_root}
train: images/train
val: images/val
test: images/test

nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
"""
    yaml_path = Path(output_root) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\ndata.yaml written to {yaml_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--visdrone_root', type=str, required=True,
                        help='Path to raw VisDrone dataset root')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Where to save the converted YOLO dataset')
    args = parser.parse_args()
    
    for split in ['train', 'val', 'test']:
        convert_split(args.visdrone_root, args.output_root, split)
    
    create_data_yaml(args.output_root)
    print("\nConversion complete.")