import os
import shutil
import random
from PIL import Image

# Config
SOURCE_DIR = "./PetImages"
BASE_DIR = "./dataset"
CLASSES = ['Cat', 'Dog']
SPLIT_SIZE = 0.85 # 85% for training, 15% for testing

# Create directories
for split in ['training', 'validation']:
    for cls in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

def split_data(source, train_dir, val_dir):
    files = []
    # Clean corrupt images
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        try:
            img = Image.open(file_path) # Try opening
            img.verify() # Verify it's an image
            if os.path.getsize(file_path) > 0:
                files.append(filename)
        except (IOError, SyntaxError):
            print(f"Skipping bad file: {filename}")

    # Shuffle and split
    random.shuffle(files)
    split_point = int(len(files) * SPLIT_SIZE)
    train_files = files[:split_point]
    val_files = files[split_point:]

    # Copy files
    for f in train_files:
        shutil.copyfile(os.path.join(source, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copyfile(os.path.join(source, f), os.path.join(val_dir, f))

# Run execution
for cls in CLASSES:
    print(f"Processing {cls} images...")
    split_data(
        os.path.join(SOURCE_DIR, cls),
        os.path.join(BASE_DIR, 'training', cls),
        os.path.join(BASE_DIR, 'validation', cls)
    )
print("Data preparation complete!")