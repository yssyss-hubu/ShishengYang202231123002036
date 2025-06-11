import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 自定义数据集类
class ImageTxtDataset(Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            img_path, label = line.strip().split()
            label = int(label)
            full_img_path = os.path.join(folder_name, img_path) if not os.path.isabs(img_path) else img_path
            self.imgs_path.append(full_img_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 简易ConvNeXt模块实现
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # 可扩展为DropPath实现，这里用Identity简化

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        # 变换形状适配 LayerNorm 和 Linear: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x

# 定义stem层，修正LayerNorm输入维度问题
class StemLayer(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(out_chans, eps=1e-6, elementwise_affine=True)

    def forward(self, x):
        x = self.conv(x)              # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)    # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)    # (B, C, H, W)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=10, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        # Stem
        self.downsample_layers = nn.ModuleList()
        stem = StemLayer(in_chans, dims[0])  # 替换为修正的StemLayer
        self.downsample_layers.append(stem)

        # Downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6, elementwise_affine=True),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # Stages (ConvNeXt blocks)
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            if i == 0:
                x = self.downsample_layers[i](x)  # 第一个Downsample是stem层，已经处理好LayerNorm
            else:
                b, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1)             # (B, H, W, C)
                x = self.downsample_layers[i][0](x)  # LayerNorm
                x = x.permute(0, 3, 1, 2)             # (B, C, H, W)
                x = self.downsample_layers[i][1](x)  # Conv2d
            x = self.stages[i](x)

        x = x.mean([-2, -1])  # 全局平均池化 (B, C)
        x = self.norm(x)      # LayerNorm
        x = self.head(x)
        return x


def main():
    # 数据处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = ImageTxtDataset(txt_path='train.txt', folder_name='images', transform=train_transform)
    test_dataset = ImageTxtDataset(txt_path='test.txt', folder_name='images', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(set(train_dataset.labels))

    # 使用自定义ConvNeXt模型
    model = ConvNeXt(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    best_model_path = 'best_convnext.pth'

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            acc = 100. * correct / total
            pbar.set_postfix({'loss': total_loss / total, 'acc': f'{acc:.2f}%'})

        # 验证
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        test_acc = 100. * correct / total
        print(f"✅ Epoch {epoch+1} Test Accuracy: {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved at epoch {epoch+1} with accuracy {test_acc:.2f}%")

    # 加载最佳模型再次评估
    print("\n🔁 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    print(f"🎯 Final Test Accuracy (Best Model): {100. * correct / total:.2f}%")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows 多线程兼容
    main()
