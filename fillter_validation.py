import os
import shutil
from pathlib import Path

# Create validation directories for class 0 only
val_images = Path("dataset/val_class0/images")
val_labels = Path("dataset/val_class0/labels")
val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

# Find all images with only class 0 labels
count = 0
for label_file in Path("dataset/val/labels").glob("*.txt"):
    with open(label_file) as f:
        lines = f.readlines()
        # Check if all lines (all detections) are class 0
        if all(int(line.split()[0]) == 0 for line in lines if line.strip()):
            # This label file has only class 0, copy it and the image
            label_src = label_file
            label_dst = val_labels / label_file.name
            
            # Find the image (it might have a different extension)
            image_src = None
            for img in Path("dataset/val/images").glob(f"{label_file.stem}*"):
                image_src = img
                break
            
            if image_src and image_src.exists():
                image_dst = val_images / image_src.name
                shutil.copy2(label_src, label_dst)
                shutil.copy2(image_src, image_dst)
                count += 1

print(f"Created validation set with {count} class-0-only images")
