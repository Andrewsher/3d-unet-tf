import tensorflow as tf


def dice(y_true, y_pred):
    TP = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 2.0 * (TP + 1) / (union + 1)

def dice_1(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]
    TP = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 2.0 * (TP + 1) / (union + 1)

def dice_2(y_true, y_pred):
    y_true = y_true[..., 2]
    y_pred = y_pred[..., 2]
    TP = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 2.0 * (TP + 1) / (union + 1)

def dice_3(y_true, y_pred):
    y_true = y_true[..., 3]
    y_pred = y_pred[..., 3]
    TP = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 2.0 * (TP + 1) / (union + 1)