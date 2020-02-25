import os

import tensorflow as tf
from sklearn.model_selection import KFold
import argparse
import numpy as np
from tqdm import tqdm

from dataset.dataset import get_T1_dataset, get_T2_dataset
from models.unet import Unet
from loss.loss import dice_loss
from loss.metrics import dice, dice_1, dice_2, dice_3

# tf.enable_eager_execution()

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-path', default='/data/data/iseg2019/training_h5/iseg2019_training.h5', type=str)
    parser.add_argument('-t', '--target-path', default='output-1', type=str)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    # Train Setting
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    args = parser.parse_args()

    return args


def train(fold, train_case_indexes, val_case_indexes, args):
    tf.keras.backend.clear_session()
    log_dir = os.path.join(args.target_path, 'fold_' + str(fold) + '/')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # get loss and op
    loss_object = dice_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model = Unet(n_class=4)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice, dice_1, dice_2, dice_3])

    train_dataset = get_T1_dataset(data_path=args.data_path, case_indexes=train_case_indexes, batch_size=args.batch_size)
    val_dataset = get_T1_dataset(data_path=args.data_path, case_indexes=val_case_indexes, batch_size=args.batch_size)

    model.fit(train_dataset,
              validation_data=val_dataset,
              steps_per_epoch=840//args.batch_size,
              validation_steps=210//args.batch_size,
              callbacks=[tensorboard])

    model.save_weights(os.path.join(log_dir, 'final_weights.h5'))
    return record_loss, record_iou_1, record_iou_2, record_iou_3, record_dice_1, record_dice_2, record_dice_3


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.target_path):
        os.mkdir(args.target_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cases_indexes = [i for i in range(10)]
    kf = KFold(n_splits=5, shuffle=False)

    loss_f, iou_1_f, iou_2_f, iou_3_f, dice_1_f, dice_2_f, dice_3_f = [], [], [], [], [], [], []

    for fold, (train_patient_indexes, val_patient_indexes) in enumerate(kf.split(cases_indexes)):
        record_loss, record_iou_1, record_iou_2, record_iou_3, record_dice_1, record_dice_2, record_dice_3 = \
            train(fold, train_patient_indexes, val_patient_indexes, args)
        loss_f.append(record_loss)
        iou_1_f.append(record_iou_1)
        iou_2_f.append(record_iou_2)
        iou_3_f.append(record_iou_3)
        dice_1_f.append(record_dice_1)
        dice_2_f.append(record_dice_2)
        dice_3_f.append(record_dice_3)

    templete = 'Loss: {}, IoU_1: {}, IoU_2: {}, IoU_3: {}, Dice_1: {}, Dice_2: {}, Dice_3: {} '
    print(templete.format(np.round(np.mean(loss_f), 5),
                          np.round(np.mean(iou_1_f), 5),
                          np.round(np.mean(iou_2_f), 5),
                          np.round(np.mean(iou_3_f), 5),
                          np.round(np.mean(dice_1_f), 5),
                          np.round(np.mean(dice_2_f), 5),
                          np.round(np.mean(dice_3_f), 5)))

