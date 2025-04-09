import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
from dataSet import VOCDataset
from model import VGGNet, FCN8s
from display import visualize_results
from metric import *
from utils import mkdir
from plot import *

mkdir("model")
mkdir("pic")

# 如果在linux, 需要设置 device
if os.name == 'posix':
    torch.cuda.set_device(7)
num_epoch = 10
batch_size = 16
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dataset = VOCDataset(root="./data", split="train")
test_dataset = VOCDataset(root="./data", split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型训练
model = FCN8s(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=21).to(device)
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def train(model, train_loader, loss_func, optimizer):
    model.train()     
    epoch_train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        outputs = F.log_softmax(outputs, dim=1)
        loss = loss_func(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()       # 梯度清零
        loss.backward()             # 反向传播求解梯度
        optimizer.step()            # 更新权重参数

        epoch_train_loss += loss.item()
    
    return epoch_train_loss

def test(model, test_loader, loss_func):
    epoch_metrics = Metrics()
    model.eval()
    epoch_test_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            outputs = F.log_softmax(outputs, dim=1)
            loss = loss_func(outputs, labels)

            batch_metric = calculate_metrics(preds, labels)
            epoch_metrics += batch_metric
            
            epoch_test_loss += loss.item()
            num_batches += 1

    return epoch_test_loss, (epoch_metrics / num_batches)


def main():
    plot_frequency = 5
    metrics_list = []
    train_losses = []
    test_losses = []

    print("Start Training...")
    for epoch in range(1, num_epoch+1):
        train_time = time.time()
        epoch_train_loss = train(model, train_loader, loss_func, optimizer)
        print(f"Epoch {epoch}/{num_epoch} \tTime: {time.time()-train_time:.4f} \tTrain Loss: {epoch_train_loss:.4f}")
        train_losses.append(epoch_train_loss)


        if epoch % plot_frequency == 0 or epoch == num_epoch:
            print("Start Testing...")
            test_time = time.time()
            epoch_test_loss, epoch_avg_metrics = test(model, test_loader, loss_func)
            print(f"Test Time: {time.time()-test_time:.4f} \tTest Loss: {epoch_test_loss:.4f}")
            print(epoch_avg_metrics)
            test_losses.append(epoch_test_loss)
            metrics_list.append(epoch_avg_metrics)


    # 保存模型
    torch.save(model.state_dict(), f"model/model_{num_epoch}.pth")
    plot_loss(train_losses, test_losses, save_path="./pic")
    plot_metrics(metrics_list, plot_frequency, num_epoch)

    visualize_results(model, train_loader, device)


if __name__ == "__main__":
    print("===================Start===================")
    print(f"TrainData_len:  \t{len(train_dataset)}")
    print(f"TestData_len:   \t{len(test_dataset)}")
    print(f"Device:         \t{device}")
    print(f"GPU_id:         \t{torch.cuda.current_device()}")
    print(f"GPU_num:        \t{torch.cuda.device_count()}")

    print(f"Epoch:          \t{num_epoch}")
    print(f"Batch_size:     \t{batch_size}")
    print(f"Learning_rate:  \t{learning_rate}")
    print(f"Model:          \t{model.__class__.__name__}")
    print(f"Loss Function:  \t{loss_func.__class__.__name__}")
    print(f"Optimizer:      \t{optimizer.__class__.__name__}")
    print("===========================================")
    
    main()