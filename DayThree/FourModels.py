import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import timm
from tqdm import tqdm

# 配置
batch_size = 64
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

# 数据增强 & 预处理（调整图像大小以适配模型）
transform_train = transforms.Compose([
    transforms.Resize(224),  # 调整为适合预训练模型的尺寸
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 均值
                         (0.2023, 0.1994, 0.2010))   # CIFAR-10 方差
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 加载数据集（使用已下载的目录）
trainset = torchvision.datasets.CIFAR10(
    root='E:\PythonProject\demo6_9\DayThree\dataset_chen', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

testset = torchvision.datasets.CIFAR10(
    root='E:\PythonProject\demo6_9\DayThree\dataset_chen', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


# 模型工厂
def get_model(name):
    if name == 'resnet':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=True, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'moganet':
        model = timm.create_model('moga_b1', pretrained=True, num_classes=num_classes)

    else:
        raise ValueError("Unknown model name")
    return model.to(device)


# 训练与评估
def train_and_eval(model_name):
    print(f"\n📢 Training model: {model_name.upper()}")
    model = get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        print(f"🔄 Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch+1}")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total_loss += loss.item() * labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                running_acc = 100. * correct / total
                print(f"  📦 Batch {batch_idx+1} - Loss: {loss.item():.4f} - Running Acc: {running_acc:.2f}%")

        epoch_acc = 100. * correct / total
        print(f"✅ Epoch {epoch+1} Summary - Avg Loss: {total_loss/total:.4f}, Train Accuracy: {epoch_acc:.2f}%\n")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="🔍 Evaluating"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    print(f"✅ {model_name.upper()} Test Accuracy: {100.*correct/total:.2f}%")


if __name__ == "__main__":
    model_list = ['resnet', 'googlenet', 'mobilenet', 'moganet']
    for name in model_list:
        train_and_eval(name)
