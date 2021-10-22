
'''
FILENAME = "train-images-idx3-ubyte"
FILEPATH = "./data/MNIST/raw" # 根据mnist.pkl.gz文件的实际保存路径修改

import pickle, gzip
# 将MNIST数据集加载到内存的标准方式
with gzip.open(FILEPATH, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-root")
print(type(x_train)) #查看数据类型

import numpy as np
import matplotlib.pyplot as plt
#显示前100张图片
for i in range(100):
    plt.subplot(10, 10, i+root)
    plt.imshow(x_train[i].reshape((28,28)), cmap="gray")
    plt.axis('off') # 关闭坐标轴

plt.show()
print(x_train.shape)
print(x_train[0].shape)
'''


import torchvision.transforms as transforms
from torchvision import utils as ut
from PIL import Image


image_path = "data/root/1/3.png"
image = Image.open(image_path)
image=image.resize((28,28))
input_transform = transforms.Compose([
    transforms.Grayscale(1), #这一句就是转为单通道灰度图像
    transforms.ToTensor(),
])
out=input_transform(image)
show=transforms.ToPILImage()
show(out).show()
ut.save_image(out, "data/root/1/123.png")
