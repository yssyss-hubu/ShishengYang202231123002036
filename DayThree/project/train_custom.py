import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
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


# 数据处理方式
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# 加载数据集
train_dataset = ImageTxtDataset(txt_path='train.txt', folder_name='images', transform=train_transform)
test_dataset = ImageTxtDataset(txt_path='test.txt', folder_name='images', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 模型配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(train_dataset.labels))  # 自动推断类别数

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练与评估
epochs = 10
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

    # 每轮训练完进行验证
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    test_acc = 100. * correct / total
    print(f"✅ Epoch {epoch+1} Test Accuracy: {test_acc:.2f}%\n")
