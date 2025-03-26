import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataSet import VOCDataset
from model import SimpleSegmentationModel

from sklearn.metrics import (
    jaccard_score,  # 用于计算IoU
    f1_score,       # 用于计算F1
    accuracy_score, # 用于计算Accuracy
    recall_score,   # 用于计算Recall
)
from scipy.spatial.distance import directed_hausdorff  # 用于计算HD
from medpy.metric.binary import dc  # 用于计算Dice系数
from metric import calculate_metrics, total_metrics

epoch = 10
batch_size = 8
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = VOCDataset(root="./data", split="train")
test_dataset = VOCDataset(root="./data", split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 模型训练
model = SimpleSegmentationModel(num_classes=21)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(model, train_loader, loss_func, optimizer):
    model.train()     # 对于特定的层会有作用，如果没有不写也行
    total_train_loss = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        output = model(img)
        loss = loss_func(output, label)

        # 优化器优化模型
        optimizer.zero_grad()       # 梯度清零
        loss.backward()             # 反向传播求解梯度
        optimizer.step()            # 更新权重参数
        
        total_train_loss += loss.item()
    
    return total_train_loss



def visualize_results(model, dataloader):
    model.eval()
    # 这里应该补全可视化代码，并且输出<原图，预测图，真实标签图>
    images, masks = next(iter(dataloader))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    
    fig, ax = plt.subplots(num_samples, 3, figsize=(10, 10))
    for i in range(num_samples):
        ax[i,0].imshow(images[i].permute(1,2,0))
        ax[i,1].imshow(preds[i])
        ax[i,2].imshow(masks[i])
    plt.show()




def test(model, test_loader, ):
    model.eval()
    num_batches = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            
            outputs = model(img)
            preds = outputs.argmax(dim=1)
            
            batch_metrics = calculate_metrics(preds, label)

            for metric_name in total_metrics.keys():
                total_metrics[metric_name] += batch_metrics[metric_name]
            
            num_batches += 1
    
    # 计算平均值
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # 打印所有指标
    print("\n评估指标:")
    print(f"mIoU: {avg_metrics['mIoU']:.4f}")
    print(f"Dice: {avg_metrics['Dice']:.4f}")
    print(f"Hausdorff Distance: {avg_metrics['HD']:.4f}")
    print(f"Accuracy: {avg_metrics['Accuracy']:.4f}")
    print(f"Recall: {avg_metrics['Recall']:.4f}")
    print(f"F1 Score: {avg_metrics['F1']:.4f}")
    
    return avg_metrics

def main():
    print("===================Start===================")
    print(f"TrainData_len:  \t{len(train_dataset)}")
    print(f"TestData_len:   \t{len(test_dataset)}")
    print(f"Device:         \t{device}")
    print(f"Epoch:          \t{epoch}")
    print(f"Batch_size:     \t{batch_size}")
    print(f"Learning_rate:  \t{learning_rate}")
    print(f"Model:          \t{model.__class__.__name__}")
    print(f"Loss Function:  \t{loss_func.__class__.__name__}")
    print(f"Optimizer:      \t{optimizer.__class__.__name__}")
    print("===========================================")

    

    
    # train_and_test(net, loss_func, optimizer)

if __name__ == "__main__":

    # 对下面进行调整，不一定需要adam，并分析不同lr对结果的影响
    # lr = 1e-3
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # model = train_model(model, dataloader)
    # visualize_results(model, dataloader)
    # avg_metrics = test_model(model, dataloader)

    main()