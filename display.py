import matplotlib.pyplot as plt
import warnings
from utils import *
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def visualize_results(model, dataloader, device, num_samples=3):
    model.eval()
    # 这里应该补全可视化代码，并且输出<原图，预测图，真实标签图>
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        preds = model(images).argmax(1)
    
    display(images.cpu(), preds.cpu(), masks.cpu(), num_samples)


'''
    imgs: tensor(cpu), shape: [batch_size, 3, H, W]
    preds: tensor(cpu), shape: [batch_size, H, W]
    label: tensor(cpu), shape: [batch_size, H, W]
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
        ax[i,1].imshow(Tensor2PIL(preds[i]))
        ax[i,2].imshow(Tensor2PIL(labels[i]))

        ax[i,1].set_title("Predict")
        ax[i,0].set_title("Image")
        ax[i,2].set_title("Label")

    plt.tight_layout()
    plt.savefig("pic/result.png")
    plt.show()





if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from model import FCNs, VGGNet
    from dataSet import VOCDataset
    from metric import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VOCDataset(root="./data", split="val")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 加载模型
    model = FCNs(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=21)
    model.load_state_dict(torch.load("model/model_5.pth"))  # 加载参数
    model.to(device)
    

    model.eval()
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        preds = model(images).argmax(1)


    metrics = calculate_metrics(preds, labels, num_classes=21)
    print(metrics)

    display(images.cpu(), preds.cpu(), labels.cpu(), num_samples=3)

