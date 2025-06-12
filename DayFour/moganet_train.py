import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from moganet import MogaNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import ImageFile

# 允许加载截断图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# 配置参数
config = {
    'data_dir': './dataset',
    'arch': 'tiny',
    'num_classes': 10,
    'drop_path_rate': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,
}

# RGB兼容ImageFolder
class RGBImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {path}, 错误: {e}")
            raise e
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# 数据加载与增强
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),  # 不使用 hue
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dir = os.path.join(config['data_dir'], 'train')
    val_dir = os.path.join(config['data_dir'], 'val')

    train_dataset = RGBImageFolder(train_dir, transform=transform_train)
    val_dataset = RGBImageFolder(val_dir, transform=transform_val)
    config['num_classes'] = len(train_dataset.classes)

    print("🔍 类别信息:", train_dataset.classes)
    print(f"📊 训练集: {len(train_dataset)} 张图像")
    print(f"📊 验证集: {len(val_dataset)} 张图像")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader

# 构建模型
def build_model():
    model = MogaNet(
        arch=config['arch'],
        num_classes=config['num_classes'],
        drop_path_rate=config['drop_path_rate'],
        attn_force_fp32=True,
        fork_feat=False
    )

    if torch.cuda.device_count() > 1:
        print(f"🚀 使用 {torch.cuda.device_count()} 个 GPU")
        model = nn.DataParallel(model)

    return model.to(config['device'])

# 单轮训练
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for images, labels in loop:
        images, labels = images.to(config['device']), labels.to(config['device'])

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

        acc = 100. * total_correct / total_samples
        loop.set_postfix(loss=total_loss / total_samples, acc=acc)

    return total_correct / total_samples * 100, total_loss / total_samples

# 验证
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    loop = tqdm(loader, desc="Validation")

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(config['device']), labels.to(config['device'])
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

            acc = 100. * total_correct / total_samples
            loop.set_postfix(loss=total_loss / total_samples, acc=acc)

    return total_correct / total_samples * 100, total_loss / total_samples

# 可视化训练过程
def plot_training(train_accs, val_accs, train_losses, val_losses):
    epochs = range(1, len(train_accs)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, 'bo-', label='Train Acc')
    plt.plot(epochs, val_accs, 'ro-', label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b--', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.title('Loss')

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

# 主函数
def main():
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_dataloaders()
    model = build_model()
    print(f"📐 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"📈 Epoch {epoch}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"moganet_{config['arch']}_best.pth")
            print(f"✅ 保存新最佳模型，验证准确率: {best_acc:.2f}%")

    plot_training(train_accs, val_accs, train_losses, val_losses)
    print(f"\n🏁 训练完成！最佳验证准确率: {best_acc:.2f}%")

# Windows 下主入口保护
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
