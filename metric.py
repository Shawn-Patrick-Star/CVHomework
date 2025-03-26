import numpy as np
from sklearn.metrics import (
    jaccard_score,  # 用于计算IoU
    f1_score,       # 用于计算F1
    accuracy_score, # 用于计算Accuracy
    recall_score,   # 用于计算Recall
)
from scipy.spatial.distance import directed_hausdorff  # 用于计算HD



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
