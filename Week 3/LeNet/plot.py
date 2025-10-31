from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

train_data = FashionMNIST(root="./data",
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):  # 取出第一批batch
    if step > 0:  # step从0开始，第一批batch提取后，break
        break
batch_x = b_x.squeeze().numpy()  # 64张图片 64*224*224
batch_y = b_y.numpy()  # 64张图片的类别 64*1
class_label = train_data.classes  # 获取类别
print(class_label)

# 可视化一个 Batch 的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)  # 创建 4×16 子图布局（共64张图）
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)  # 显示第 ii 张灰度图像
    plt.title(class_label[batch_y[ii]], size=10)  # 显示类别名称
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()
