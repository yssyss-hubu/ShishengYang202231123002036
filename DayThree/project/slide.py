import os
import shutil
import random
from tqdm import tqdm


def split_dataset(original_dir='images', output_dir='dataset', train_ratio=0.8):
    classes = os.listdir(original_dir)
    for cls in classes:
        class_path = os.path.join(original_dir, cls)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        train_len = int(len(images) * train_ratio)
        train_imgs = images[:train_len]
        val_imgs = images[train_len:]

        for phase, image_list in zip(['train', 'val'], [train_imgs, val_imgs]):
            save_dir = os.path.join(output_dir, phase, cls)
            os.makedirs(save_dir, exist_ok=True)
            for img in tqdm(image_list, desc=f"Copying {cls} -> {phase}"):
                src = os.path.join(class_path, img)
                dst = os.path.join(save_dir, img)
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    split_dataset(original_dir='images', output_dir='dataset', train_ratio=0.8)
