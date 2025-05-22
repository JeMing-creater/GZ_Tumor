import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
def top_k_accuracy(output, target, k=1):
    """计算Top-K准确率"""
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / target.size(0)).item()


def calculate_f1_score(outputs, targets, threshold=0.5):
    """计算F1分数"""
    predictions = (outputs > threshold).int().cpu().numpy().flatten()
    true_labels = targets.int().cpu().numpy().flatten()
    return f1_score(true_labels, predictions)

def specificity(outputs, targets, threshold=0.5):
    """计算特异性"""
    predictions = (outputs > threshold).int().cpu().numpy().flatten()
    true_labels = targets.int().cpu().numpy().flatten()
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0

def quadratic_weighted_kappa(outputs, targets, threshold=0.5):
    """计算二次加权Kappa"""
    predictions = (outputs > threshold).int().cpu().numpy().flatten()
    true_labels = targets.int().cpu().numpy().flatten()
    return cohen_kappa_score(true_labels, predictions, weights='quadratic')


def calculate_metrics(output, target, accelerator):
    """计算批次的统计量"""
    predictions = output.int().cpu().numpy().flatten()
    true_labels = target.int().cpu().numpy().flatten()
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    return {'tp': torch.tensor(tp).to(accelerator.device), 'tn': torch.tensor(tn).to(accelerator.device), 'fp': torch.tensor(fp).to(accelerator.device), 'fn': torch.tensor(fn).to(accelerator.device)}

def accumulate_metrics(batch_metrics, accelerator, accumulated_metrics=None):
    """累积所有批次的指标"""
    if accumulated_metrics is None:
        accumulated_metrics = {'tp': torch.tensor(0).to(accelerator.device), 'tn': torch.tensor(0).to(accelerator.device), 'fp': torch.tensor(0).to(accelerator.device), 'fn': torch.tensor(0).to(accelerator.device)}
    
    for key in ['tp', 'tn', 'fp', 'fn']:
        accumulated_metrics[key] += batch_metrics[key]
    
    return accumulated_metrics

def compute_final_metrics(accumulated_metrics):
    """根据累积的统计数据计算最终的指标"""
    tp, tn, fp, fn = float(accumulated_metrics['tp']), float(accumulated_metrics['tn']), float(accumulated_metrics['fp']), float(accumulated_metrics['fn'])
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
    }