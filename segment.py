import torch
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







# 请不要使用torchvision的VOCSegmentation，独立实现dataset以及dataloader
def get_dataloader(batch_size=8):
    # 注意这里只有train的dataset，在测试时候请实现test的dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    # 独立实现dataset的构建
    dataset = VOCDataset(root="./data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


def train_model(model, dataloader, criterion, optimizer, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        # 补全训练代码，需要计算loss
        for images, masks in dataloader:
            pass
    return model

def visualize_results(model, dataloader):
    model.eval()
    # 这里应该补全可视化代码，并且输出<原图，预测图，真实标签图>
    # with torch.no_grad():
        # pass

def calculate_metrics(pred, target, num_classes=21):
    """
    计算多个分割评估指标 mIoU, Dice, HD, Accuracy, Recall, F1
    """
    metrics = {}
    
    # 将输入转换为numpy数组
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    # 计算mIoU
    def mIoU():
        return jaccard_score(target, pred, average='macro', 
                           labels=range(num_classes), 
                           zero_division=0)
    
    # 计算Dice系数
    def dice_score():
        return f1_score(target, pred, average='macro',  # Dice系数等价于F1 score
                       labels=range(num_classes),
                       zero_division=0)
    
    # 计算Hausdorff距离
    def hausdorff_distance():
        scores = []
        pred_2d = pred.reshape(224, 224)
        target_2d = target.reshape(224, 224)
        for i in range(num_classes):
            pred_mask = (pred_2d == i)
            target_mask = (target_2d == i)
            if not np.any(pred_mask) or not np.any(target_mask):
                scores.append(0)
                continue
            pred_points = np.array(np.where(pred_mask)).T
            target_points = np.array(np.where(target_mask)).T
            scores.append(max(directed_hausdorff(pred_points, target_points)[0],
                            directed_hausdorff(target_points, pred_points)[0]))
        return np.mean(scores)

    # 计算Accuracy
    def accuracy():
        return accuracy_score(target, pred)

    # 计算Recall
    def recall():
        return recall_score(target, pred, average='macro',
                          labels=range(num_classes),
                          zero_division=0)

    # 计算F1 score
    def f1():
        return f1_score(target, pred, average='macro',
                       labels=range(num_classes),
                       zero_division=0)
    
    # 计算所有指标
    metrics['mIoU'] = mIoU()
    metrics['Dice'] = dice_score()
    metrics['HD'] = hausdorff_distance()
    metrics['Accuracy'] = accuracy()
    metrics['Recall'] = recall()
    metrics['F1'] = f1()
    
    return metrics

def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 初始化所有指标的累加器
    total_metrics = {
        'mIoU': 0,
        'Dice': 0,
        'HD': 0,
        'Accuracy': 0,
        'Recall': 0,
        'F1': 0
    }
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            batch_metrics = calculate_metrics(preds, masks)

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

if __name__ == "__main__":
    dataloader = get_dataloader()
    model = SimpleSegmentationModel(num_classes=21)
    # 对下面进行调整，不一定需要adam，并分析不同lr对结果的影响
    # lr = 1e-3
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    model = train_model(model, dataloader)
    visualize_results(model, dataloader)
    avg_metrics = test_model(model, dataloader)