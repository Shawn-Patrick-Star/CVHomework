import torch
import numpy as np
import os
from PIL import Image

COLORMAP = {
    (0, 0, 0): 0,         # Background
    (128, 0, 0): 1,       # Aeroplane
    (0, 128, 0): 2,       # Bicycle
    (128, 128, 0): 3,     # Bird
    (0, 0, 128): 4,       # Boat
    (128, 0, 128): 5,     # Bottle
    (0, 128, 128): 6,     # Bus
    (128, 128, 128): 7,   # Car
    (64, 0, 0): 8,        # Cat
    (192, 0, 0): 9,       # Chair
    (64, 128, 0): 10,     # Cow
    (192, 128, 0): 11,    # Dining table
    (64, 0, 128): 12,     # Dog
    (192, 0, 128): 13,    # Horse
    (64, 128, 128): 14,   # Motorbike
    (192, 128, 128): 15,  # Person
    (0, 64, 0): 16,       # Potted plant
    (128, 64, 0): 17,     # Sheep
    (0, 192, 0): 18,      # Sofa
    (128, 192, 0): 19,    # Train
    (0, 64, 128): 20,     # TV/Monitor
}
colormap2label = np.zeros(256 ** 3, dtype=np.uint8)
for i, colormap in enumerate(COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i




def PIL2Tensor(label_pil):
    """
    convert label (PIL image) to label (int64 tensor).
    """
    label_np = np.array(label_pil, dtype=np.int32)
    idx = (label_np[:, :, 0] * 256 + label_np[:, :, 1]) * 256 + label_np[:, :, 2]
    return torch.tensor(colormap2label[idx], dtype=torch.int64)
    
def Tensor2PIL(label_tensor):
    """
    convert label (int64 tensor) to label (PIL image).
    """
    label_np = np.zeros((label_tensor.shape[0], label_tensor.shape[1], 3), dtype=np.uint8)
    for i, colormap in enumerate(COLORMAP):
        label_np[label_tensor == i] = colormap
    return Image.fromarray(label_np, mode='RGB')



def denormalize(tensor):
    # 反归一化处理
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor.clone()
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)