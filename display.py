import matplotlib.pyplot as plt
from utils import denormalize

'''
    imgs: tensor, shape: [batch_size, 3, H, W]
    preds: tensor, shape: [batch_size, H, W]
    label: tensor, shape: [batch_size, H, W]
'''
def display(imgs, preds, labels, num_samples=3):

    fig, ax = plt.subplots(num_samples, 3, figsize=(8, 8))


    for i in range(num_samples):
        # pred 和 label是索引图，需要转换为PIL图像
  
        '''
        PyTorch输入格式 → 网络要求 [通道数, 高, 宽]
        Matplotlib可视化 → 需要 [高, 宽, 通道数]
        故使用 permute 
        '''
        ax[i,0].imshow(denormalize(imgs[i]).permute(1,2,0))
        ax[i,1].imshow(preds[i])
        ax[i,2].imshow(labels[i])

        ax[i,1].set_title("Predict")
        ax[i,0].set_title("Image")
        ax[i,2].set_title("Label")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch
    from model import LightSegNet
    from torch.utils.data import DataLoader
    from dataSet import VOCDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = LightSegNet()  # 实例化模型结构
    model.load_state_dict(torch.load("model.pth"))  # 加载参数
    model.to(device)
    
    model.eval()
    # 这里应该补全可视化代码，并且输出<原图，预测图，真实标签图>
    dataset = VOCDataset(root="./data", split="val")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        preds = model(images).argmax(1)
    
    print(images[0].shape, preds[0].shape, masks[0].shape)

    display(images.cpu(), preds.cpu(), masks.cpu(), num_samples=3)

