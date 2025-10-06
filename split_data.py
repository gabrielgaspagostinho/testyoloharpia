import os
import random
import shutil

IMAGES_SOURCE_DIR = 'imagens_originais'
LABELS_SOURCE_DIR = 'anotacoes_temporarias'
DATASET_DIR = 'datasets'
TRAIN_RATIO = 0.8

os.makedirs(os.path.join(DATASET_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'labels/val'), exist_ok=True)

all_images = [f for f in os.listdir(IMAGES_SOURCE_DIR) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

split_index = int(len(all_images) * TRAIN_RATIO)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

def copy_files(file_list, dest_folder):
    for filename in file_list:
        basename, _ = os.path.splitext(filename)
        shutil.copy(os.path.join(IMAGES_SOURCE_DIR, filename), os.path.join(DATASET_DIR, f'images/{dest_folder}', filename))
        shutil.copy(os.path.join(LABELS_SOURCE_DIR, f'{basename}.txt'), os.path.join(DATASET_DIR, f'labels/{dest_folder}', f'{basename}.txt'))

copy_files(train_images, 'train')
copy_files(val_images, 'val')
print(f"Divisão concluída!")