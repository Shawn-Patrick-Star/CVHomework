import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    jaccard_score,  # 用于计算IoU
    f1_score,       # 用于计算F1
    accuracy_score, # 用于计算Accuracy
    recall_score,   # 用于计算Recall
)
from scipy.spatial.distance import directed_hausdorff  # 用于计算HD
from medpy.metric.binary import dc  # 用于计算Dice系数


class Metrics:
    """
    计算多个分割评估指标 mIoU, Dice, HD, Accuracy, Recall, F1
    """
    def __init__(self, mIoU=0, Dice=0, HD=0, Accuracy=0, Recall=0, F1=0):
        self.mIoU = mIoU
        self.Dice = Dice
        self.HD = HD
        self.Accuracy = Accuracy
        self.Recall = Recall
        self.F1 = F1

    def __add__(self, other):
        return Metrics(
            mIoU=self.mIoU + other.mIoU,
            Dice=self.Dice + other.Dice,
            HD=self.HD + other.HD,
            Accuracy=self.Accuracy + other.Accuracy,
            Recall=self.Recall + other.Recall,
            F1=self.F1 + other.F1
        )

    def __truediv__(self, num):
        return Metrics(
            mIoU=self.mIoU / num,
            Dice=self.Dice / num,
            HD=self.HD / num,
            Accuracy=self.Accuracy / num,
            Recall=self.Recall / num,
            F1=self.F1 / num
        )
    
    def __str__(self):
        return (f"mIoU:      \t{self.mIoU:.4f}\n"
                f"Dice:      \t{self.Dice:.4f}\n"
                f"HD:        \t{self.HD:.4f}\n"
                f"Accuracy:  \t{self.Accuracy:.4f}\n"
                f"Recall:    \t{self.Recall:.4f}\n"
                f"F1:        \t{self.F1:.4f}")


# 初始化所有指标的累加器
total_metrics = {
    'mIoU':     0,      # mean Intersection over Union 平均交并比 
    'Dice':     0,      # Dice coefficient 骰子系数 
    'HD':       0,      # Hausdorff Distance 豪斯多夫距离
    'Accuracy': 0,      
    'Recall':   0,      # 
    'F1':       0       # F1 score 
}


def calculate_metrics(pred, target, num_classes=21):
    """
    计算多个分割评估指标 mIoU, Dice, HD, Accuracy, Recall, F1
    
    Args:
        pred:   tensor(gpu) [B, H, W] 或 [B*H*W]
        target: tensor(gpu) [B, H, W] 或 [B*H*W]
        num_classes: 类别数量
    
    Returns:
        metrics: 包含各种评估指标的字典
    """
    
    # 确保输入是正确的形状
    if pred.dim() > 2:
        batch_size, h, w = pred.shape
    else:
        # 如果输入是展平的，需要恢复原始形状
        batch_size = 1
        h = w = int(np.sqrt(len(pred)))
    
    # 将输入转换为numpy数组
    pred_np = pred.cpu().numpy().reshape(batch_size, h, w)
    target_np = target.cpu().numpy().reshape(batch_size, h, w)
    
    # 计算mIoU
    def mIoU():
        # 只考虑实际出现在标签中的类别
        present_classes = np.unique(target_np)
        present_classes = present_classes[present_classes < num_classes]
        
        if len(present_classes) == 0:
            return 0.0
            
        return jaccard_score(target_np.flatten(), pred_np.flatten(), average='macro', 
                           labels=present_classes, 
                           zero_division=0)
    
    # 计算Dice系数
    def dice_score():
        # 只考虑实际出现在标签中的类别
        present_classes = np.unique(target_np)
        present_classes = present_classes[present_classes < num_classes]
        
        if len(present_classes) == 0:
            return 0.0
            
        return f1_score(target_np.flatten(), pred_np.flatten(), average='macro',  # Dice系数等价于F1 score
                       labels=present_classes,
                       zero_division=0)
    
    # 计算Hausdorff距离
    def hausdorff_distance():
        scores = []
        
        # 逐个处理批量中的每个图像
        for i in range(batch_size):
            pred_2d = pred_np[i]
            target_2d = target_np[i]
            
            # 只计算实际出现的类别
            present_classes = np.unique(np.concatenate([np.unique(pred_2d), np.unique(target_2d)]))
            present_classes = present_classes[present_classes < num_classes]
            
            for c in present_classes:
                pred_mask = (pred_2d == c)
                target_mask = (target_2d == c)
                
                # 如果某个类别在预测或真实标签中不存在，跳过
                if not np.any(pred_mask) or not np.any(target_mask):
                    continue
                
                try:
                    # 提取边界点
                    pred_contours = np.array(np.where(pred_mask ^ ndimage.binary_erosion(pred_mask))).T
                    target_contours = np.array(np.where(target_mask ^ ndimage.binary_erosion(target_mask))).T
                    
                    # 如果某个类别的边界为空，跳过
                    if len(pred_contours) == 0 or len(target_contours) == 0:
                        continue
                    
                    # 计算双向Hausdorff距离
                    forward_hd = directed_hausdorff(pred_contours, target_contours)[0]
                    backward_hd = directed_hausdorff(target_contours, pred_contours)[0]
                    scores.append(max(forward_hd, backward_hd))
                except Exception as e:
                    print(f"Error calculating HD for class {c} in image {i}: {e}")
                    continue
        
        # 如果没有计算任何距离，返回0
        if len(scores) == 0:
            return 0.0
            
        return np.mean(scores)

    # 计算Accuracy
    def accuracy():
        return accuracy_score(target_np.flatten(), pred_np.flatten())

    # 计算Recall
    def recall():
        # 只考虑实际出现在标签中的类别
        present_classes = np.unique(target_np)
        present_classes = present_classes[present_classes < num_classes]
        
        if len(present_classes) == 0:
            return 0.0
            
        return recall_score(target_np.flatten(), pred_np.flatten(), average='macro',
                          labels=present_classes,
                          zero_division=0)

    # 计算F1 score
    def f1():
        # 只考虑实际出现在标签中的类别
        present_classes = np.unique(target_np)
        present_classes = present_classes[present_classes < num_classes]
        
        if len(present_classes) == 0:
            return 0.0
            
        return f1_score(target_np.flatten(), pred_np.flatten(), average='macro',
                       labels=present_classes,
                       zero_division=0)
    
    # 计算所有指标
    try:
        metrics = Metrics(
            mIoU=mIoU(),
            Dice=dice_score(),
            HD=hausdorff_distance(),
            Accuracy=accuracy(),
            Recall=recall(),
            F1=f1()
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")

    return metrics

if __name__ == "__main__":
    import torch

    # 输入形状 [B=2, H=3, W=3]
    pred = torch.tensor([
        [[0,1,2], [2,1,0], [0,1,2]],
        [[2,1,0], [0,1,2], [2,1,0]]
    ], dtype=torch.long)

    target = pred.clone() # 完美匹配

    metrics = calculate_metrics(pred, target, num_classes=3)
    # 所有指标应达到理论最优值
    assert metrics.mIoU == 1.0
    assert metrics.Dice == 1.0
    assert metrics.Accuracy == 1.0
    assert metrics.Recall == 1.0
    print("Test1 Passed!")




    # 非正方形的图像形状 [B=1, H=4, W=6]
    pred = torch.randint(0, 3, (1,4,6))
    target = pred.clone() # 强制预测正确

    metrics = calculate_metrics(pred, target)
    # 展平后与3D输入计算逻辑应一致
    assert metrics.mIoU == 1.0
    assert metrics.Dice == 1.0
    assert metrics.Accuracy == 1.0
    print("Test2 Passed!")





    # 输入形状 [B=2, H=2, W=2]
    pred = torch.tensor([
        [[1,1], [1,1]],  # 全预测为1
        [[0,0], [0,0]]   # 全预测为0
    ], dtype=torch.long)

    target = torch.tensor([
        [[1,1], [0,0]],  # 第一个batch有2个错误
        [[0,0], [0,0]]   # 第二个batch全对
    ], dtype=torch.long)

    metrics = calculate_metrics(pred, target, num_classes=2)
    print(metrics)
    # 全局计算指标：
    # 第一个batch：TP=2, FP=2, FN=2 → IoU=2/(2+2+2)=0.333
    # 第二个batch：TP=4, FP=0, FN=0 → IoU=1.0
    # mIoU = (0.333 + 1.0)/2 = 0.666...
    assert round(metrics.mIoU, 2) == 0.67
    assert metrics.Accuracy == (2+4)/(4+4) == 0.75 # (正确像素数6/总像素数8)
    print("Test3 Passed!")