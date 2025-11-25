import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 读取图像
Input_path = 'snapshots/result_F'
save_path = 'result_F'
Paths = os.listdir(Input_path)

for path in Paths:
    img_path = os.path.join(Input_path, path)
    print(img_path)
    image = Image.open(img_path)

    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 提取图像的RGB通道
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # 绘制RGB通道直方图重叠在一幅图上
    # range (0,255) -> (5,250)
    plt.hist(red_channel.ravel(), bins=253, range=(2,254), density=True, color='red', alpha=0.5, label='Red')
    plt.hist(green_channel.ravel(), bins=253, range=(2,254), density=True, color='green', alpha=0.5, label='Green')
    plt.hist(blue_channel.ravel(), bins=253, range=(2,254), density=True, color='blue', alpha=0.5, label='Blue')

    plt.ylim(0, 0.07)
    # 设置标题和轴标签
    # plt.title('RGB Channel Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    # 显示图例
    plt.legend()

    # 显示图像和直方图
    save_path_f = './snapshots/Hist/' + save_path
    if not os.path.exists(save_path_f):
        os.makedirs(save_path_f)
    plt.savefig(save_path_f + '/' + path)
    plt.close()
