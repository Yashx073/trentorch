import numpy as np

def precision_recall_f1(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, float]:
    """
    labels, predictions: {0, 1}-valued, same shape.

    Returns:
        (precision, recall, f1). Each is 0.0 if its denominator would
        be 0 (no positive predictions for precision, no actual
        positives for recall).
    """
    # TODO: true_positives  = predicted 1 AND actually 1
    #       false_positives = predicted 1 AND actually 0
    #       false_negatives = predicted 0 AND actually 1
    # precision = tp / (tp + fp), recall = tp / (tp + fn)
    # f1 = 2 * precision * recall / (precision + recall)
    n_samples = len(labels)
    TP = np.sum((labels == 1) & (predictions == 1))
    FP = np.sum((labels == 0) & (predictions == 1))
    FN = np.sum((labels == 1) & (predictions == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    labels: {0, 1}-valued.
    scores: a continuous score (e.g. a predicted probability), higher
    means "more likely class 1."

    Returns:
        (fpr, tpr, thresholds), one entry per threshold tried:
        every distinct score value, from highest to lowest, plus
        +infinity (the threshold that classifies everything as 0).
        fpr[i]/tpr[i] are the false/true positive rate when predicting
        1 for every score >= thresholds[i].
    """
    # TODO: thresholds = [inf] followed by every unique score, sorted
    # descending. For each threshold, predictions = (scores >=
    # threshold), then the same tp/fp counting as precision_recall_f1,
    # divided by the TOTAL number of actual positives/negatives (not
    # predicted) to get tpr/fpr.
    thresholds = np.concatenate([[np.inf], np.sort(np.unique(scores))[::-1]])     
    n_positive = np.sum(labels == 1)     
    n_negative = np.sum(labels == 0)      
    
    tpr_list, fpr_list = [], []     
    for threshold in thresholds:         
        predictions = (scores >= threshold).astype(int)         
        true_positives = np.sum((predictions == 1) & (labels == 1))         
        false_positives = np.sum((predictions == 1) & (labels == 0))         
        
        tpr_list.append(true_positives / n_positive if n_positive > 0 else 0.0)         
        fpr_list.append(false_positives / n_negative if n_negative > 0 else 0.0)      
        
    return np.array(fpr_list), np.array(tpr_list), thresholds




def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve, via the trapezoidal rule."""
    # TODO: Sort by fpr first (roc_curve's own order is descending by
    # threshold, not ascending by fpr), then np.trapezoid. Use a stable sort:
    # many points share the same fpr, and their order changes the area.
    order = np.argsort(fpr, kind = "stable")
    return float(np.trapezoid(tpr[order], fpr[order]))
