import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 输入3通道，输出96通道，11x11卷积，步长4，padding=2，输出尺寸缩小
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第二层卷积，输出256通道，5x5卷积，padding=2保持尺寸
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层卷积，输出384通道，3x3卷积，padding=1保持尺寸
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层卷积，输出384通道，3x3卷积，padding=1保持尺寸
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第五层卷积，输出256通道，3x3卷积，padding=1保持尺寸
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 计算特征图尺寸
        # 输入224x224 -> conv1(11x11, s=4, p=2): (224+2*2-11)/4+1=55
        # maxpool1(3x3, s=2): floor((55-3)/2+1)=27
        # conv2(5x5, p=2): 27
        # maxpool2(3x3, s=2): floor((27-3)/2+1)=13
        # conv3,padding=1: 13
        # conv4,padding=1: 13
        # conv5,padding=1: 13
        # maxpool3(3x3, s=2): floor((13-3)/2+1)=6

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    model = AlexNet(num_classes=10)
    input_tensor = torch.randn(4, 3, 224, 224)
    output = model(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)


if __name__ == "__main__":
    main()
