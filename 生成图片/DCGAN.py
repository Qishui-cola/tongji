import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os

# 数据归一化(-1,1)
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-1
    transforms.Normalize(0.5, 0.5)  # 均值0.5方差0.5
])
# 加载内置数据集
train_ds = torchvision.datasets.MNIST('data',
                                      train=True,
                                      transform=transform,
                                      download=True)

dataloader = torch.utils.data.DataLoader(train_ds,
                                         batch_size=64,
                                         shuffle=True)

# 定义生成器，依然输入长度100的噪声
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256*7*7)
        self.bn1 = nn.BatchNorm1d(256*7*7)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1
                                          )
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1
                                          )
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1
                                          )

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.deconv1(x))
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.tanh(self.deconv3(x))
        return x


# 判别器,输入（28，28）图片
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*6*6, 1)

    def forward(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1(x)), p=0.3)
        x = F.dropout2d(F.leaky_relu(self.conv2(x)), p=0.3)
        x = self.bn(x)
        x = x.view(-1, 128*6*6)
        x = torch.sigmoid(self.fc(x))
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('using cuda:', torch.cuda.get_device_name(0))

Gen = Generator().to(device)
Dis = Discriminator().to(device)

loss_fun = nn.BCELoss()
d_optimizer = torch.optim.Adam(Dis.parameters(), lr=1e-5)  # 小技巧
g_optimizer = torch.optim.Adam(Gen.parameters(), lr=1e-4)

# 创建保存图片的文件夹
output_dir = "./result/DCGAN"
os.makedirs(output_dir, exist_ok=True)

def generate_and_save_image(model, test_input):
    predictions = np.squeeze(model(test_input).cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i]+1) / 2, cmap='gray')
        plt.axis('off')
    #plt.show()
    # 保存图片
    save_path = os.path.join(output_dir, f"epoch_{epoch + 1}.png")
    plt.savefig(save_path)
    plt.close(fig)

test_input = torch.randn(16, 100, device=device)
D_loss = []
G_loss = []

for epoch in range(300):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader.dataset)
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optimizer.zero_grad()
        real_output = Dis(img)  # 判别器输入真实图片
        # 判别器在真实图像上的损失
        d_real_loss = loss_fun(real_output,
                                    torch.ones_like(real_output)
                                    )
        d_real_loss.backward()

        gen_img = Gen(random_noise)
        fake_output = Dis(gen_img.detach())  # 判别器输入生成图片,fake_output对生成图片的预测
        # gen_img是由生成器得来的，但我们现在只对判别器更新，所以要截断对Gen的更新
        # detach()得到了没有梯度的tensor，求导到这里就停止了，backward的时候就不会求导到Gen了

        d_fake_loss = loss_fun(fake_output,
                                    torch.zeros_like(fake_output)
                                    )
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.step()

        # 更新生成器
        g_optimizer.zero_grad()
        fake_output = Dis(gen_img)
        g_loss = loss_fun(fake_output,
                               torch.ones_like(fake_output))
        g_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

    with torch.no_grad():  # 之后的内容不进行梯度的计算（图的构建）
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:', epoch+1)

        if (epoch + 1) % 10 == 0:
            generate_and_save_image(model=Gen, test_input=test_input)

plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.legend()
plt.show()


