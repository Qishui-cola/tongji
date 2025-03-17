import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
from torch.nn.functional import cosine_similarity
import torch.nn as nn

# 定义MNIST分类器 (以MNISTClassifier为例)
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 提取128维特征
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x, extract_features=False):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if extract_features:
            x = self.fc[0](x)  # 提取128维特征
        else:
            x = self.fc(x)
        return x


# 加载MNIST分类器
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = MNISTClassifier().to(device)
classifier.load_state_dict(torch.load('mnist_classifier.pth'))  # 加载预训练权重
classifier.eval()

# 数据变换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为单通道
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集 (用于比较)
mnist_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=64, shuffle=False)

# 加载自定义图片
custom_image = Image.open('./result/GAN/epoch_100.png')  # 替换为你的图片路径
custom_image_tensor = transform(custom_image).unsqueeze(0).to(device)  # 添加batch维度

# 提取自定义图片的特征
with torch.no_grad():
    custom_features = classifier(custom_image_tensor, extract_features=True)

# 计算与MNIST样本的相似度
similarities = []
with torch.no_grad():
    for images, _ in mnist_loader:
        images = images.to(device)
        mnist_features = classifier(images, extract_features=True)
        sim = cosine_similarity(custom_features, mnist_features, dim=1)
        similarities.extend(sim.cpu().numpy())

# 获取最相似的MNIST样本
max_similarity = max(similarities)
most_similar_index = np.argmax(similarities)

print(f"与MNIST样本的最高相似度: {max_similarity}")
print(f"最相似的MNIST样本索引: {most_similar_index}")
