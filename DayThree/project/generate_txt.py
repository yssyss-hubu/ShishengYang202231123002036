import os
import random


def generate_dataset_txt(image_root='images', train_ratio=0.8, output_dir='.'):
    class_names = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    train_lines = []
    test_lines = []

    for cls_name in class_names:
        cls_path = os.path.join(image_root, cls_name)
        image_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 检查文件是否存在
        image_files = [f for f in image_files if os.path.isfile(os.path.join(cls_path, f))]

        full_paths = [os.path.join(cls_name, f) for f in image_files]
        random.shuffle(full_paths)

        split_idx = int(len(full_paths) * train_ratio)
        train_images = full_paths[:split_idx]
        test_images = full_paths[split_idx:]

        label = class_to_label[cls_name]
        train_lines += [f"{img_path} {label}\n" for img_path in train_images]
        test_lines += [f"{img_path} {label}\n" for img_path in test_images]

    # 写入 train.txt 和 test.txt
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.writelines(train_lines)
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.writelines(test_lines)

    print("✅ 成功生成 train.txt 和 test.txt")
    print("📦 类别标签映射:", class_to_label)


if __name__ == "__main__":
    generate_dataset_txt(image_root='images', train_ratio=0.8, output_dir='.')
