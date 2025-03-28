import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import Tensor2PIL, Tensor2PIL


'''
    images: tensor, shape: [batch_size, 3, H, W]
    preds: tensor, shape: [batch_size, H, W]
    label: tensor, shape: [batch_size, H, W]
'''
def display(images, preds, labels, num_samples=3):

    fig, ax = plt.subplots(num_samples, 3, figsize=(8, 8))


    for i in range(num_samples):
        # pred 和 label是索引图，需要转换为PIL图像
  
        '''
        PyTorch输入格式 → 网络要求 [通道数, 高, 宽]
        Matplotlib可视化 → 需要 [高, 宽, 通道数]
        故使用 permute 
        '''
        ax[i,0].imshow(images[i].permute(1,2,0))
        pred = Tensor2PIL(preds[i])
        ax[i,1].imshow(pred)

        label = Tensor2PIL(labels[i])
        ax[i,2].imshow(label)

        ax[i,1].set_title("Predict")
        ax[i,0].set_title("Image")
        ax[i,2].set_title("Label")

    plt.tight_layout()
    plt.show()

