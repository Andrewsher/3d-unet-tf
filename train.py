import os

import tensorflow as tf
from sklearn.model_selection import KFold
import argparse
import numpy as np

from dataset.dataset import get_T1_dataset, get_T2_dataset, get_dataset_hdr
from models.unet import Unet
from loss.loss import dice_loss
from loss.metrics import dice

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


def grad(model, inputs, targets, loss):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss_value = loss(targets, predictions)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# @tf.function
def train_step(model, inputs, targets, loss_func, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        train_loss = loss_func(targets, predictions)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train(fold, train_case_indexes, val_case_indexes, args):
    tf.keras.backend.clear_session()
    log_dir = os.path.join(args.target_path, 'fold_' + str(fold) + '/')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model = Unet(n_class=4)

    # get loss and op
    loss_object = dice_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # get metrics
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    iou_1 = tf.keras.metrics.MeanIoU(num_classes=2, name='train_iou_1')
    iou_2 = tf.keras.metrics.MeanIoU(num_classes=2, name='train_iou_2')
    iou_3 = tf.keras.metrics.MeanIoU(num_classes=2, name='train_iou_3')
    dice_1 = tf.keras.metrics.Mean(name='train_dice_1')
    dice_2 = tf.keras.metrics.Mean(name='train_dice_2')
    dice_3 = tf.keras.metrics.Mean(name='train_dice_3')

    step = 0

    record_loss, record_iou_1, record_iou_2, record_iou_3, record_dice_1, record_dice_2, record_dice_3 = 0,0,0,0,0,0,0

    summary_writer = tf.summary.create_file_writer(log_dir, model._graph)

    train_dataset = get_dataset_hdr(args.data_path, case_indexes=train_case_indexes, batch_size=args.batch_size, is_training=True)
    val_dataset = get_dataset_hdr(args.data_path, case_indexes=val_case_indexes, batch_size=args.batch_size, is_training=False)

    template_1 = 'Epoch {}, Loss: {}, IoU_1: {}, IoU_2: {}, IoU_3: {}, Dice_1: {}, Dice_2: {}, Dice_3: {} '
    template_2 = 'Val Loss: {}, Val_IoU_1: {}, Val_IoU_2: {}, Val_IoU_3: {}, Val_Dice_1: {}, Val_Dice_2: {}, Val_Dice_3: {} '
    print('-----------------------------------------------------------')
    print('fold ', fold)

    lr = args.lr
    for epoch in range(args.epochs):
        if (epoch + 1) % 5 == 0:
            lr = lr * 0.2
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # training
        loss_metric.reset_states()
        iou_1.reset_states()
        iou_2.reset_states()
        iou_3.reset_states()
        dice_1.reset_states()
        dice_2.reset_states()
        dice_3.reset_states()
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                train_loss = loss_object(labels, predictions)
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


            loss_metric.update_state(train_loss)
            iou_1.update_state(labels[..., 1], predictions[..., 1])
            iou_2.update_state(labels[..., 2], predictions[..., 2])
            iou_3.update_state(labels[..., 3], predictions[..., 3])
            dice_1.update_state(dice(labels[:, :, :, :, 1], predictions[:, :, :, :, 1]))
            dice_2.update_state(dice(labels[:, :, :, :, 2], predictions[:, :, :, :, 2]))
            dice_3.update_state(dice(labels[:, :, :, :, 3], predictions[:, :, :, :, 3]))
        print(template_1.format(epoch + 1,
                                np.round(loss_metric.result().numpy(), 5),
                                np.round(iou_1.result().numpy() * 100, 5),
                                np.round(iou_2.result().numpy() * 100, 5),
                                np.round(iou_3.result().numpy() * 100, 5),
                                np.round(dice_1.result().numpy() * 100, 5),
                                np.round(dice_2.result().numpy() * 100, 5),
                                np.round(dice_3.result().numpy() * 100, 5)))
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', loss_metric.result(), step=epoch)
            tf.summary.scalar('train_iou_1', iou_1.result(), step=epoch)
            tf.summary.scalar('train_iou_2', iou_2.result(), step=epoch)
            tf.summary.scalar('train_iou_3', iou_3.result(), step=epoch)
            tf.summary.scalar('train_dice_1', dice_1.result(), step=epoch)
            tf.summary.scalar('train_dice_2', dice_2.result(), step=epoch)
            tf.summary.scalar('train_dice_3', dice_3.result(), step=epoch)
            tf.summary.image('prediction', predictions[:, :, :, 32, :], step=epoch)
            tf.summary.image('label', labels[:, :, :, 32, :], step=epoch)

        # validation
        loss_metric.reset_states()
        iou_1.reset_states()
        iou_2.reset_states()
        iou_3.reset_states()
        dice_1.reset_states()
        dice_2.reset_states()
        dice_3.reset_states()
        for images, labels in val_dataset:
            predictions = model(images)
            val_loss = loss_object(labels, predictions)
            loss_metric.update_state(val_loss)
            iou_1.update_state(labels[..., 1], predictions[..., 1])
            iou_2.update_state(labels[..., 2], predictions[..., 2])
            iou_3.update_state(labels[..., 3], predictions[..., 3])
            dice_1.update_state(dice(labels[:, :, :, :, 1], predictions[:, :, :, :, 1]))
            dice_2.update_state(dice(labels[:, :, :, :, 2], predictions[:, :, :, :, 2]))
            dice_3.update_state(dice(labels[:, :, :, :, 3], predictions[:, :, :, :, 3]))

        print(template_2.format(np.round(loss_metric.result().numpy(), 5),
                                np.round(iou_1.result().numpy() * 100, 5),
                                np.round(iou_2.result().numpy() * 100, 5),
                                np.round(iou_3.result().numpy() * 100, 5),
                                np.round(dice_1.result().numpy() * 100, 5),
                                np.round(dice_2.result().numpy() * 100, 5),
                                np.round(dice_3.result().numpy() * 100, 5)))
        with summary_writer.as_default():
            tf.summary.scalar('val_loss', loss_metric.result(), step=epoch)
            tf.summary.scalar('val_iou_1', iou_1.result(), step=epoch)
            tf.summary.scalar('val_iou_2', iou_2.result(), step=epoch)
            tf.summary.scalar('val_iou_3', iou_3.result(), step=epoch)
            tf.summary.scalar('val_dice_1', dice_1.result(), step=epoch)
            tf.summary.scalar('val_dice_2', dice_2.result(), step=epoch)
            tf.summary.scalar('val_dice_3', dice_3.result(), step=epoch)
            tf.summary.image('val_prediction', predictions[:, :, :, 32, :], step=epoch)
            tf.summary.image('val_label', labels[:, :, :, 32, :], step=epoch)

        record_loss, record_iou_1, record_iou_2, record_iou_3, record_dice_1, record_dice_2, record_dice_3 = \
            loss_metric.result().numpy(), \
            iou_1.result().numpy() * 100, \
            iou_2.result().numpy() * 100,\
            iou_3.result().numpy() * 100, \
            dice_1.result().numpy() * 100,\
            dice_2.result().numpy() * 100, \
            dice_3.result().numpy() * 100,

        model.save_weights(os.path.join(log_dir, 'ep=' + str(epoch) + '.h5'))

    model.save_weights(os.path.join(log_dir, 'final_weights.h5'))
    return record_loss, record_iou_1, record_iou_2, record_iou_3, record_dice_1, record_dice_2, record_dice_3


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.target_path):
        os.mkdir(args.target_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cases_indexes = np.array([i for i in range(1, 11, 1)])
    kf = KFold(n_splits=5, shuffle=False)

    loss_f, iou_1_f, iou_2_f, iou_3_f, dice_1_f, dice_2_f, dice_3_f = [], [], [], [], [], [], []

    for fold, (train_patient_indexes, val_patient_indexes) in enumerate(kf.split(cases_indexes)):
        print(cases_indexes[train_patient_indexes])
        print(cases_indexes[val_patient_indexes])
        record_loss, record_iou_1, record_iou_2, record_iou_3, record_dice_1, record_dice_2, record_dice_3 = \
            train(fold, cases_indexes[train_patient_indexes], cases_indexes[val_patient_indexes], args)
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

