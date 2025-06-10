import torch
import torchvision
from PIL import Image

from model import *
from torchvision import transforms

classes = [
    'airplane',    # 0
    'automobile',  # 1
    'bird',        # 2
    'cat',         # 3
    'deer',        # 4
    'dog',         # 5
    'frog',        # 6
    'horse',       # 7
    'ship',        # 8
    'truck'        # 9
]


image_path = "./Image/img.png"
image = Image.open(image_path)
print(image)
# png格式是四个通道，除了RGB三个通道外，还有一个透明通道,调用image = image.convert('RGB')
image = image.convert('RGB')

# 改变成tensor格式
trans_reszie = transforms.Resize((32, 32))
trans_totensor = transforms.ToTensor()
transform = transforms.Compose([trans_reszie, trans_totensor])
image = transform(image)
print(image.shape)

# 加载训练模型
model = torch.load("model_save\\chen_9.pth").to("cuda")

# print(model)

image = torch.reshape(image, (1, 3, 32, 32)).to("cuda")
# image = image.cuda()

# 将模型转换为测试模型
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

# 推理预测
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(1).item()


# 输出结果
print(f"预测类别编号：{predicted_class}")
print(f"预测类别名称（英文）：{classes[predicted_class]}")
