import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score, precision_recall_fscore_support

def compute_segmentation_metrics(preds, targets, num_classes=7, ignore_index=6):
    """
    Computes diverse segmentation metrics ignoring `ignore_index`.
    Args:
        preds: 1D array or tensor of predictions
        targets: 1D array or tensor of ground truth labels
    Returns:
        dict of metrics
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    mask = targets != ignore_index
    preds_valid = preds[mask]
    targets_valid = targets[mask]
    
    if len(targets_valid) == 0:
        return {
            'OA': 0.0,
            'AA': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_iou': 0.0,
            'mIoU': 0.0,
            'class_precision': {},
            'class_recall': {},
            'class_f1': {},
            'class_iou': {},
            'class_support': {},
            'mcc': 0.0,
            'kappa': 0.0,
            'confusion_matrix': np.zeros((num_classes, num_classes)).tolist()
        }
        
    # Valid classes
    labels = [i for i in range(num_classes) if i != ignore_index]
    
    # F1 scores
    macro_f1 = f1_score(targets_valid, preds_valid, labels=labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(targets_valid, preds_valid, labels=labels, average='weighted', zero_division=0)
    precision_arr, recall_arr, class_f1_arr, support_arr = precision_recall_fscore_support(
        targets_valid,
        preds_valid,
        labels=labels,
        zero_division=0,
    )
    
    class_precision = {lbl: value for lbl, value in zip(labels, precision_arr)}
    class_recall = {lbl: value for lbl, value in zip(labels, recall_arr)}
    class_f1 = {lbl: f1 for lbl, f1 in zip(labels, class_f1_arr)}
    class_support = {lbl: int(value) for lbl, value in zip(labels, support_arr)}
    
    # Confusion matrix
    cm = confusion_matrix(targets_valid, preds_valid, labels=labels)
    
    # IoU
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    iou_arr = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    class_iou = {lbl: iou for lbl, iou in zip(labels, iou_arr)}
    macro_iou = np.mean(iou_arr)
    oa = float(np.mean(preds_valid == targets_valid))
    aa = float(np.mean(recall_arr))
    
    # MCC and Kappa
    mcc = matthews_corrcoef(targets_valid, preds_valid)
    kappa = cohen_kappa_score(targets_valid, preds_valid)
    
    return {
        'OA': oa,
        'AA': aa,
        'macro_f1': float(macro_f1),
        'macro_F1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_iou': float(macro_iou),
        'mIoU': float(macro_iou),
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'class_iou': class_iou,
        'class_support': class_support,
        'mcc': float(mcc),
        'kappa': float(kappa),
        'confusion_matrix': cm.tolist()
    }
