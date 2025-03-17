import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from PIL import Image
import csv
import pandas as pd


classes = ('real', 'fake')  # fake 1 real 0

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)
path='./testdata'
testList=os.listdir(path)


with open('cla_pre.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for file in testList:
        img=Image.open(path+file).convert('RGB')    # 改动
        # img=Image.fromarray(fft_lbp_rgb(path+file))
        img=transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        out=model(img)
        # Predict
        _, pred = torch.max(out.data, 1)
        # print('{} : {}'.format(file,classes[pred.data.item()]),file=f)
        if(file[-3:]=="jpg"):
            writer.writerow([file[:-4], 1 - pred.data.item()])
        else:
            writer.writerow([file[:-5], 1 - pred.data.item()])

# 读取CSV文件
df = pd.read_csv('cla_pre.csv')

# 按第一列字典序升序排序
df_sorted = df.sort_values(by=df.columns[0], ascending=True)

# 保存排序后的CSV文件
df_sorted.to_csv('sorted_file.csv', index=False)

print("文件已按第一列字典序升序排序并保存为 sorted_file.csv")