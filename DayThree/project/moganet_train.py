import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from moganet import MogaNet
from tqdm import tqdm

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
    'epochs': 5,
    'lr': 0.001,  # 降低学习率，适合小数据集
    'weight_decay': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,
}

# 加载数据集
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
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

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    config['num_classes'] = len(train_dataset.classes)

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
        print(f"使用 {torch.cuda.device_count()} 个 GPU")
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

    return acc

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

    return acc

# 主训练函数
def main():
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_dataloaders()
    model = build_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"moganet_{config['arch']}_best.pth")
            print(f"保存最佳模型，验证准确率: {best_acc:.2f}%")

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")

# Windows 下主入口保护
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
