import os
import torch
import torchvision.transforms as transforms
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.transform = transform
        self.split = split
        
        self.img_dir = os.path.join(root, "JPEGImages")
        self.label_dir = os.path.join(root, "SegmentationClassVOC21")

        # 根据split读取对应的文件名列表（示例需要实际文件列表）
        split_file = os.path.join(root, "ImageSets/Segmentation", split + ".txt")
        with open(split_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]
        
        self.image_files = [os.path.join(self.img_dir, f"{name}.jpg") for name in file_names]
        self.label_files = [os.path.join(self.label_dir, f"{name}.png") for name in file_names]
        
        # 标签转换：Resize使用最近邻插值
        self.label_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        # 图像转换：Resize使用双线性插值
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        label = Image.open(self.label_files[idx])

        img = self.image_transform(img)
        label = self.label_transform(label).squeeze(0).long()

        return img, label
