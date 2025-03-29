import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from utils import PIL2Tensor


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, crop_size=(224, 224)):
        self.root = root
        self.split = split
        self.crop_size = crop_size

        self.image_files, self.label_files = get_fileList(root, split=="train")
        self.image_files = self.filter(self.image_files)
        self.label_files = self.filter(self.label_files)
        
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def filter(self, imgs):  # 过滤掉尺寸小于crop_size的图片
        return [img for img in imgs if (
                Image.open(img).size[1] >= self.crop_size[0] and
                Image.open(img).size[0] >= self.crop_size[1])]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_pil = Image.open(self.image_files[idx]).convert('RGB')
        label_pil = Image.open(self.label_files[idx]).convert('RGB')  # 读取标签图像

        img_pil, label_pil = rand_crop(img_pil, label_pil, *self.crop_size)

        img_tensor = self.image_transform(img_pil)
        label_tensor = PIL2Tensor(label_pil)

        return img_tensor, label_tensor


def get_fileList(root, is_train=True): 
    img_dir = os.path.join(root, "JPEGImages")
    label_dir = os.path.join(root, "SegmentationClass")

    # 根据split读取对应的文件名列表（示例需要实际文件列表）
    split = "train" if is_train else "val"
    split_file = os.path.join(root, "ImageSets/Segmentation", split + ".txt")
    with open(split_file, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    
    image_files = [os.path.join(img_dir, f"{name}.jpg") for name in file_names]
    label_files = [os.path.join(label_dir, f"{name}.png") for name in file_names]

    return image_files, label_files

def rand_crop(image, label, height, width):
    """
    Random crop image (PIL image) and label (PIL image).
    """
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(height, width))

    image = transforms.functional.crop(image, i, j, h, w)
    label = transforms.functional.crop(label, i, j, h, w)

    return image, label


if __name__ == "__main__":

    from display import display
    from torch.utils.data import DataLoader
    import numpy as np

    dataset = VOCDataset(root="./data", split="train")
    img, label = dataset[100]
    # print(img.shape)
    # print(img.dtype)
    # print(img)
    # print()
    # print(label.shape)
    # print(label.dtype)
    # print(label)

    # print(np.unique(label.numpy()))
    

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    imgs, labels = next(iter(dataloader))
    print(np.unique(labels[0].numpy()))
    

    display(imgs, labels, labels, num_samples=3)
     