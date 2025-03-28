import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from dataSet import VOCDataset
from model import LightSegNet
from display import display
from metric import calculate_metrics, total_metrics

num_epoch = 1
batch_size = 16
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = VOCDataset(root="./data", split="train")
test_dataset = VOCDataset(root="./data", split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 模型训练
model = LightSegNet(num_classes=21).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(model, train_loader, loss_func, optimizer):
    model.train()     
    total_train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        output = model(imgs)
        loss = loss_func(output, labels)

        # 优化器优化模型
        optimizer.zero_grad()       # 梯度清零
        loss.backward()             # 反向传播求解梯度
        optimizer.step()            # 更新权重参数
        
        total_train_loss += loss.item()
    
    return total_train_loss

def test(model, test_loader):
    model.eval()
    total_test_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            loss = loss_func(outputs, labels)
            
            print(preds.shape, labels.shape)

            for label, pred in zip(preds, labels): # 逐样本计算指标
                print(label.shape, pred.shape)
                calculate_metrics(pred, label)
            
            total_test_loss += loss.item()
            num_batches += 1
    
    # 计算平均值
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_metrics, total_test_loss

def train_and_test(model, loss_func, optimizer):
    
    for epoch in range(1, num_epoch+1):

        start_time = time.time()
        total_train_loss = train(model, train_loader, loss_func, optimizer)
        print(f"Epoch {epoch}/{num_epoch} \tTime: {time.time() - start_time} \tTrain Loss: {total_train_loss:.4f}")

    # visualize_results(model, train_loader)

    avg_metrics, total_test_loss = test(model, test_loader)
    # 打印所有指标
    print(f"\n--------- Avg_Metrics----------:")
    print(f"mIoU:               \t{avg_metrics['mIoU']:.4f}")
    print(f"Dice:               \t{avg_metrics['Dice']:.4f}")
    print(f"Hausdorff Distance: \t{avg_metrics['HD']:.4f}")
    print(f"Accuracy:           \t{avg_metrics['Accuracy']:.4f}")
    print(f"Recall:             \t{avg_metrics['Recall']:.4f}")
    print(f"F1 Score:           \t{avg_metrics['F1']:.4f}")

def visualize_results(model, dataloader, num_samples=3):
    model.eval()
    # 这里应该补全可视化代码，并且输出<原图，预测图，真实标签图>
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)


    with torch.no_grad():
        preds = model(images).argmax(1)
    

    display(images.cpu(), preds.cpu(), masks.cpu(), num_samples)



def main():
    print("===================Start===================")
    print(f"TrainData_len:  \t{len(train_dataset)}")
    print(f"TestData_len:   \t{len(test_dataset)}")
    print(f"Device:         \t{device}")
    print(f"Epoch:          \t{num_epoch}")
    print(f"Batch_size:     \t{batch_size}")
    print(f"Learning_rate:  \t{learning_rate}")
    print(f"Model:          \t{model.__class__.__name__}")
    print(f"Loss Function:  \t{loss_func.__class__.__name__}")
    print(f"Optimizer:      \t{optimizer.__class__.__name__}")
    print("===========================================")

    train_and_test(model, loss_func, optimizer)
    # visualize_results(model, test_loader)

if __name__ == "__main__":
    main()