from sklearn.metrics import balanced_accuracy_score
import torch
import numpy as np


def get_iou_obj(pc_preds: torch.LongTensor, targets: torch.LongTensor, label: int = 1):
    """
    Calculates the Intersection over Union (IoU) for a specific label between predicted and target point clouds 
    using PyTorch tensors.

    Args:
        pc_preds (torch.LongTensor): Predicted point cloud labels. Should be a PyTorch tensor of long integers.
        targets (torch.LongTensor): Ground truth point cloud labels. Should be a PyTorch tensor of long integers.
        label (int, optional): The label for which the IoU is calculated. Defaults to 1.

    Returns:
        float: IoU value for the specified label. Returns NaN if division by zero occurs.
    """
    # get IoU
    corrects = pc_preds == targets
    gt_positive = torch.sum(targets == label).item()
    detected_positive = (pc_preds == label).sum().item()
    tp = (corrects & (pc_preds == label)).sum().item()
    fp = detected_positive - tp

    # Handle division by zero
    if gt_positive + fp == 0:
        iou_obj = torch.nan
    else:
        iou_obj = tp / (gt_positive + fp)
    return iou_obj


def get_iou_np(pc_preds, targets, label = 1):
    """
    Calculates the Intersection over Union (IoU) for a specific label between predicted and target point clouds.

    Args:
        pc_preds (numpy.ndarray): Predicted point cloud labels. Should be a NumPy array.
        targets (numpy.ndarray): Ground truth point cloud labels. Should be a NumPy array.
        label (int, optional): The label for which the IoU is calculated. Defaults to 1.

    Returns:
        float: IoU value for the specified label. Returns NaN if division by zero occurs or IoU is 0.
    """
    # get IoU
    corrects = pc_preds == targets
    gt_positive = np.sum(targets == label)
    detected_positive = np.count_nonzero(pc_preds == label)
    tp = np.count_nonzero(corrects & (pc_preds == label))
    fp = detected_positive - tp

    # Handle division by zero
    if gt_positive + fp == 0:
        iou_obj = np.nan
    else:
        iou_obj = tp / (gt_positive + fp)

    if iou_obj == 0.:
        iou_obj = np.nan

    return iou_obj*100


def get_accuracy(preds, targets, metrics):
    corrects = torch.eq(preds.view(-1), targets.view(-1))
    if len(corrects) != 0:
        metrics['accuracy'] = (corrects.sum().item() / len(corrects))
    else:
        metrics['accuracy'] = 0
    return metrics


def get_weighted_accuracy(preds, targets, metrics, c_weights):

    sample_weights = get_weights4sample(c_weights, targets.view(-1))
    accuracy_w = balanced_accuracy_score(targets, preds, sample_weight=sample_weights)
    metrics['accuracy_w'] = accuracy_w

    return metrics


def get_weights_effective_num_of_samples(n_of_classes, beta, samples_per_cls):
    """The authors suggest experimenting with different beta values: 0.9, 0.99, 0.999, 0.9999."""
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights4class = (1.0 - beta) / np.array(effective_num)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_inverse_num_of_samples(n_of_classes, samples_per_cls, power=1.0):
    weights4class = 1.0 / np.array(np.power(samples_per_cls, power))  # [0.03724195 0.00244003]
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_sklearn(n_of_classes, samples_per_cls):
    weights4class = np.sum(samples_per_cls) / np.multiply(n_of_classes, samples_per_cls)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights4class(weighing_method, n_classes, samples_per_cls, beta=None):
    """

       :param weighing_method: str, options available: "EFS" "INS" "ISNS"
       :param n_classes: int, representing the total number of classes in the entire train set
       :param samples_per_cls: A python list of size [n_classes]
       :param beta: float,

       :return weights4class: torch.Tensor of size [batch, n_classes]
    """
    if weighing_method == 'EFS':
        weights4class = get_weights_effective_num_of_samples(n_classes, beta, samples_per_cls)
    elif weighing_method == 'INS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls)
    elif weighing_method == 'ISNS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls, 0.5)  # [0.9385, 0.0615]
    elif weighing_method == 'sklearn':
        weights4class = get_weights_sklearn(n_classes, samples_per_cls)
    else:
        return None

    weights4class = torch.tensor(weights4class).float()
    return weights4class


def get_weights4sample(weights4class, labels):
    """
    :param weights4class: Torch Tensor of size [n_classes]
    :param labels: Torch Long Tensor of size [batch * n points]

    :return:
    """
    # one-hot encoding
    labels = labels.to('cpu').numpy()  # labels defines columns of non-zero elements
    one_hot = np.zeros((labels.size, 2))  # [batch, 2]
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1

    weights4samples = weights4class.to('cpu').unsqueeze(0)
    weights4samples = weights4samples.repeat(labels.shape[0], 1)
    weights4samples = torch.tensor(np.array(weights4samples * one_hot))
    weights4samples = weights4samples.sum(1).cpu()

    return weights4samples
