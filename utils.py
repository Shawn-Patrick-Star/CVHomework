import torch
import numpy as np

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


def PIL2Tensor(label):
    """
    convert label (PIL image) to label (int64 tensor).
    """
    label = np.array(label)
    x = np.zeros(label.shape[:2], dtype=np.int64)
    for rgb, idx in COLORMAP.items():
        x[(label == np.array(rgb)).all(axis=-1)] = idx
        
    return torch.from_numpy(x)
    

def Tensor2PIL(label):
    """
    convert label (int64 tensor) to label (PIL image).
    """
    label = label.numpy()
    x = np.zeros((label.shape[0], label.shape[1], 3))
    for rgb, idx in COLORMAP.items():
        x[label == idx] = rgb

    return x

