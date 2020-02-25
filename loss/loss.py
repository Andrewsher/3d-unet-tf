import tensorflow as tf
from loss.metrics import dice


def dice_loss(y_true, y_pred, n_classes=4):
    tmp_dice = dice(y_true[..., 0], y_pred[..., 0])
    for i in range(1, n_classes):
        tmp_dice = tmp_dice + dice(y_true[..., i], y_pred[..., i])
    return 1. - (tmp_dice * 1.0 / n_classes)
