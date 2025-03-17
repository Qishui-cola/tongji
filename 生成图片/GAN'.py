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

# 返回一个批次的数据
imgs, _ = next(iter(dataloader))


# 生成器，输入100噪声输出（1，28，28）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 28, 28)
        return x

#  辨别器,输入（1，28，28），输出真假,推荐使用LeakRelu
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.linear(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('using cuda:', torch.cuda.get_device_name(0))

Gen = Generator().to(device)
Dis = Discriminator().to(device)

d_optim = torch.optim.Adam(Dis.parameters(), lr=0.001)
g_optim = torch.optim.Adam(Gen.parameters(), lr=0.001)


# BCEWithLogisticLoss 未激活的输出
loss_function = torch.nn.BCELoss()

# 创建保存图片的文件夹
output_dir = "./result/GAN"
os.makedirs(output_dir, exist_ok=True)

def gen_img_plot(model, test_input):
    # squeeze 删除单维度
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(prediction.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i]+1) / 2)  # 生成-1，1，恢复到0，1
        plt.axis('off')
        # 保存图片

    save_path = os.path.join(output_dir, f"epoch_{epoch + 1}.png")
    plt.savefig(save_path)
    plt.close(fig)
    #plt.show()


test_input = torch.randn(16, 100, device=device)
D_loss = []
G_loss = []

for epoch in range(300):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()
        real_output = Dis(img)  # 判别器输入真实图片
        # 判别器在真实图像上的损失
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output)
                                    )
        d_real_loss.backward()

        gen_img = Gen(random_noise)
        fake_output = Dis(gen_img.detach())  # 判别器输入生成图片,fake_output对生成图片的预测
        # gen_img是由生成器得来的，但我们现在只对判别器更新，所以要截断对Gen的更新
        # detach()得到了没有梯度的tensor，求导到这里就停止了，backward的时候就不会求导到Gen了

        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output)
                                    )
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        # 更新生成器        g_optim.zero_grad()
        fake_output = Dis(gen_img)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():  # 之后的内容不进行梯度的计算（图的构建）
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:', epoch+1)

        if (epoch + 1) % 10 == 0:
            gen_img_plot(model=Gen, test_input=test_input)

plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.legend()
plt.show()