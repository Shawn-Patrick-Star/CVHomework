import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
from dataSet import VOCDataset
from model import VGGNet, FCNs
from display import visualize_results
from metric import calculate_metrics, print_avg_metrics


# 如果在linux, 需要设置 device
if os.name == 'posix':
    torch.cuda.set_device(7)
num_epoch = 5
batch_size = 16
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dataset = VOCDataset(root="./data", split="train")
test_dataset = VOCDataset(root="./data", split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型训练
model = FCNs(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=21).to(device)
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(model, train_loader, loss_func, optimizer):
    model.train()     
    total_train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        outputs = F.log_softmax(outputs, dim=1)
        loss = loss_func(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()       # 梯度清零
        loss.backward()             # 反向传播求解梯度
        optimizer.step()            # 更新权重参数

        total_train_loss += loss.item()
    
    return total_train_loss

def test(model, test_loader, loss_func):
    

    model.eval()
    total_test_loss = 0
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
            
            total_test_loss += loss.item()
            num_batches += 1

    return total_test_loss


def main():
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

    print("Start Training...")
    for epoch in range(1, num_epoch+1):
        start_time = time.time()
        total_train_loss = train(model, train_loader, loss_func, optimizer)
        print(f"Epoch {epoch}/{num_epoch} \tTime: {time.time() - start_time:.4f} \tTrain Loss: {total_train_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"model/model_{num_epoch}.pth")

    print("Start Testing...")
    total_test_loss = test(model, test_loader, loss_func)
    # 打印所有指标
    print_avg_metrics(len(test_dataset))

    visualize_results(model, train_loader)


if __name__ == "__main__":
    main()