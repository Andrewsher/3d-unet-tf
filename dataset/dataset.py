import tensorflow as tf
import h5py
import numpy as np
from medpy.io import load
from tqdm import tqdm

import os


def get_T1_dataset(data_path, case_indexes, is_training=True, batch_size=8):
    f = h5py.File(data_path, 'r')
    patch_per_case = f['patch_per_case'][0]
    patch_indexes = []
    for case_index in case_indexes:
        patch_indexes.extend([patch_per_case * case_index + i for i in range(patch_per_case)])
    images = np.array(f['T1'][patch_indexes][:, :, :, :, None] / 255., dtype=np.float32)
    labels = f['label'][patch_indexes]
    labels = tf.keras.utils.to_categorical(labels, num_classes=4)
    dataset = None
    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(10000).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset


def get_T2_dataset(data_path, case_indexes, is_training=True, batch_size=8):
    f = h5py.File(data_path, 'r')
    patch_per_case = f['patch_per_case'][0]
    patch_indexes = []
    for case_index in case_indexes:
        patch_indexes.extend([patch_per_case * case_index + i for i in range(patch_per_case)])
    images = np.array(f['T2'][patch_indexes][:, :, :, :, None] / 255., dtype=np.float32)
    labels = f['label'][patch_indexes]
    labels = tf.keras.utils.to_categorical(labels, num_classes=4)
    dataset = None
    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(10000).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset


def get_dataset_hdr(data_path, case_indexes, is_training=True, batch_size=8, mode='T1', patch_size=64, step_size=32):
    images = []
    labels = []

    assert mode=='T1' or mode=='T2'

    print('Loading data')
    for i in tqdm(case_indexes, ascii=True, position=0, leave=True):
        subject_name = 'subject-%d-' % i
        f_image = os.path.join(data_path, subject_name + mode + '.hdr')
        f_label = os.path.join(data_path, subject_name + 'label.hdr')
        img_image, header_image = load(f_image)
        img_label, header_label = load(f_label)
        for idx_x in range(0, img_label.shape[0] - patch_size + 1, step_size):
            for idx_y in range(0, img_label.shape[1] - patch_size + 1, step_size):
                for idx_z in range(0, img_label.shape[2] - patch_size + 1, step_size):
                    images.append(img_image[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size, idx_z:idx_z + patch_size])

                    labels.append(img_label[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size, idx_z:idx_z + patch_size])

    images = np.array(np.array(images)[:, :, :, :, None] / 255., dtype=np.float32)
    labels = tf.keras.utils.to_categorical(np.array(labels), num_classes=4)

    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(10000).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)
    return dataset

