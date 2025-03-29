import numpy as np
from sklearn.metrics import (
    jaccard_score,  # 用于计算IoU
    f1_score,       # 用于计算F1
    accuracy_score, # 用于计算Accuracy
    recall_score,   # 用于计算Recall
)
from scipy.spatial.distance import directed_hausdorff  # 用于计算HD
from medpy.metric.binary import dc  # 用于计算Dice系数


# 初始化所有指标的累加器
total_metrics = {
    'mIoU':     0,      # mean Intersection over Union 平均交并比 
    'Dice':     0,      # Dice coefficient 骰子系数 
    'HD':       0,      # Hausdorff Distance 豪斯多夫距离
    'Accuracy': 0,      
    'Recall':   0,      # 
    'F1':       0       # F1 score 
}

def init_metrics():
    global total_metrics
    total_metrics = {k: 0 for k in total_metrics.keys()}

def print_avg_metrics(num_batches):
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    print(f"\n--------- Avg_Metrics----------:")
    print_metrics(avg_metrics)

def print_metrics(metrics):
    print(f"mIoU:               \t{metrics['mIoU']:.4f}")
    print(f"Dice:               \t{metrics['Dice']:.4f}")
    print(f"Hausdorff Distance: \t{metrics['HD']:.4f}")
    print(f"Accuracy:           \t{metrics['Accuracy']:.4f}")
    print(f"Recall:             \t{metrics['Recall']:.4f}")
    print(f"F1 Score:           \t{metrics['F1']:.4f}")
    init_metrics()



def calculate_metrics(label, pred, num_classes=21):
    """
    计算多个分割评估指标 mIoU, Dice, HD, Accuracy, Recall, F1
    """
    
    # 将输入转换为numpy数组
    pred = pred.view(-1).cpu().numpy()
    label = label.view(-1).cpu().numpy()
    
    # 计算mIoU
    def mIoU():
        return jaccard_score(label, pred, average='macro', 
                           labels=range(num_classes), 
                           zero_division=0)
    
    # 计算Dice系数
    def dice_score():
        return f1_score(label, pred, average='macro',  # Dice系数等价于F1 score
                       labels=range(num_classes),
                       zero_division=0)
    
    # 计算Hausdorff距离
    def hausdorff_distance():
        scores = []
        pred_2d = pred.reshape(224, 224)
        label_2d = label.reshape(224, 224)
        max_distance = np.sqrt(pred_2d.shape[0]**2 + pred_2d.shape[1]**2)
    
        for i in range(num_classes):
            pred_mask = (pred == i)
            label_mask = (label == i)
            
            # 处理两个掩码均为空的情况
            if not np.any(pred_mask) and not np.any(label_mask):
                scores.append(0)
                continue
                
            # 仅一方为空，返回最大距离
            if not np.any(pred_mask) or not np.any(label_mask):
                scores.append(max_distance)
                continue
                
            # 提取坐标点
            pred_points = np.stack(np.where(pred_mask), axis=1)
            label_points = np.stack(np.where(label_mask), axis=1)
            
            # 计算双向Hausdorff距离
            dh1 = directed_hausdorff(pred_points, label_points)[0]
            dh2 = directed_hausdorff(label_points, pred_points)[0]
            scores.append(max(dh1, dh2))
            
        return np.mean(scores)

    # 计算Accuracy
    def accuracy():
        return accuracy_score(label, pred)

    # 计算Recall
    def recall():
        return recall_score(label, pred, average='macro',
                          labels=range(num_classes),
                          zero_division=0)

    # 计算F1 score
    def f1():
        return f1_score(label, pred, average='macro',
                       labels=range(num_classes),
                       zero_division=0)
    
    # 计算所有指标
    total_metrics['mIoU'] +=        mIoU()
    total_metrics['Dice'] +=        dice_score()
    # total_metrics['HD'] +=          hausdorff_distance()
    total_metrics['Accuracy'] +=    accuracy()
    total_metrics['Recall'] +=      recall()
    total_metrics['F1'] +=          f1()
    

if __name__ == "__main__":
    import torch

    label = torch.zeros(224*224, dtype=torch.long)
    pred = torch.zeros(224*224, dtype=torch.long)
    calculate_metrics(label, pred, num_classes=2)
    print("测试用例1 - 相同标签和预测:")
    print_metrics(total_metrics)  # 预期所有指标为1 豪斯多夫距离为0

    # 测试用例2: 标签全0，预测中最后一个像素为1
    pred = torch.zeros(224*224, dtype=torch.long)
    pred[-1] = 1  # 修改最后一个像素
    calculate_metrics(label, pred, num_classes=2)
    print("\n测试用例2 - 单个像素差异:")
    print_metrics(total_metrics) 


    # 测试用例3: 不同区域的类别（例如类别1在上下半部分）
    label = torch.zeros(224*224, dtype=torch.long)
    label[:112*224] = 1  # 上半部分为1
    pred = torch.zeros(224*224, dtype=torch.long)
    pred[112*224:] = 1    # 下半部分为1
    calculate_metrics(label, pred, num_classes=2)
    print("\n测试用例3 - 不同区域分布:")
    print_metrics(total_metrics)

