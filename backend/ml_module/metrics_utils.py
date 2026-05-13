
import numpy as np
from scipy.integrate import trapezoid
from typing import Dict, Tuple, List


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 100) -> Tuple[List[float], List[float], float]:
    y_true = y_true.astype(int)
    
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1], 0.5
    
    tps = np.concatenate([[0], np.cumsum(y_sorted)])
    fps = np.concatenate([[0], np.arange(1, len(y_sorted) + 1) - tps[1:]])
    
    tpr_curve = tps / n_pos
    fpr_curve = fps / n_neg
    
    auc = float(trapezoid(tpr_curve, fpr_curve))
    
    if len(fpr_curve) > n_points:
        indices = np.linspace(0, len(fpr_curve) - 1, n_points, dtype=int)
        fpr_curve = fpr_curve[indices]
        tpr_curve = tpr_curve[indices]
    
    return fpr_curve.tolist(), tpr_curve.tolist(), auc


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 100) -> Tuple[List[float], List[float], float]:
    y_true = y_true.astype(int)
    
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    
    n_pos = y_sorted.sum()
    
    if n_pos == 0:
        return [0, 1], [1, 0], 0.0
    
    tps = np.concatenate([[0], np.cumsum(y_sorted)])
    fps = np.concatenate([[0], np.arange(1, len(y_sorted) + 1) - tps[1:]])
    
    recall = tps / n_pos
    precision = tps / (tps + fps)
    precision[0] = 1.0
    
    pr_auc = float(trapezoid(precision[::-1], recall[::-1]))
    
    if len(recall) > n_points:
        indices = np.linspace(0, len(recall) - 1, n_points, dtype=int)
        recall = recall[indices]
        precision = precision[indices]
    
    return recall.tolist(), precision.tolist(), pr_auc


def compute_calibration_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> Tuple[List[float], List[float]]:
    y_true = y_true.astype(int)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    frac_pos = []
    
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() > 0:
            bin_means.append(float(y_score[mask].mean()))
            frac_pos.append(float(y_true[mask].mean()))
    
    return bin_means, frac_pos


def compute_prediction_distribution(y_score: np.ndarray, y_true: np.ndarray, n_bins: int = 50) -> Dict:
    
    bins = np.linspace(0, 1, n_bins + 1)
    
    hist_neg, _ = np.histogram(y_score[y_true == 0], bins=bins)
    hist_pos, _ = np.histogram(y_score[y_true == 1], bins=bins)
    
    return {
        "bins": bins.tolist(),
        "hist_neutral": hist_neg.tolist(),
        "hist_affected": hist_pos.tolist(),
    }


def compute_metrics_vs_threshold(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 100) -> Dict:
    y_true = y_true.astype(int)
    
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        
        acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        accuracies.append(float(acc))
        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))
    
    return {
        "thresholds": thresholds.tolist(),
        "accuracy": accuracies,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
    }


def compute_confidence_vs_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> Dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    confidence = np.where(y_score >= 0.5, y_score, 1 - y_score)
    correct = (y_pred == y_true)
    
    conf_bins = np.linspace(0.5, 1.0, n_bins + 1)
    conf_levels = []
    accuracies = []
    
    for lo, hi in zip(conf_bins[:-1], conf_bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            conf_levels.append(float((lo + hi) / 2))
            accuracies.append(float(correct[mask].mean()))
    
    return {
        "confidence": conf_levels,
        "accuracy": accuracies,
    }
