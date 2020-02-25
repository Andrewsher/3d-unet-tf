import os

import tensorflow as tf
import argparse
import numpy as np
from tqdm import tqdm
from medpy.io import load, save
import math

from models.unet import Unet


def get_case(data_path, case, mode='T1', patch_size=64):
    assert mode=='T1' or mode=='T2'
    subject_name = 'subject-%d-' % case
    f_image = os.path.join(data_path, subject_name + mode + '.hdr')
    img_image, header_image = load(f_image)
    step = patch_size // 2
    l_margin = math.floor((patch_size - step) / 2)
    r_margin = math.ceil((patch_size - step) / 2)

    row, col, dim = img_image.shape[0], img_image.shape[1], img_image.shape[2]
    yield img_image.shape[0], img_image.shape[1], img_image.shape[2]
    yield math.ceil(row / step), math.ceil(col / step), math.ceil(dim / step)
    yield img_image / 255.

    top = l_margin
    bottom = math.ceil((row) / step) * step - (row) + l_margin
    left = l_margin
    right = math.ceil((col) / step) * step - (col) + l_margin
    shallow = l_margin
    deep = math.ceil((dim) / step) * step - (dim) + l_margin
    print('padding...')
    img = np.zeros(shape=(img_image.shape[0]+top+bottom, img_image.shape[1]+left+right, img_image.shape[2]+shallow+deep),
                   dtype=np.float32)
    img[top: top+img_image.shape[0], left: left+img_image.shape[1], shallow: shallow+img_image.shape[2]] = img_image[:, :, :]
    y = []
    for i in range(math.ceil(row / step)):
        for j in range(math.ceil(col / step)):
            for k in range(math.ceil(dim / step)):
                if len(y) >= 32:
                    yield np.array(y) / 255.
                    y = []

                y_tmp = np.array(img[i * step:i * step + patch_size, j * step:j * step + patch_size, k * step: k * step + patch_size])
                y.append(np.array(y_tmp)[:, :, :, None])

    if len(y) != 0:
        yield np.array(y)


def test_one_case(args, case):
    f = get_case(args.data_path, case)
    row, col, dim = f.__next__()
    img_y = np.zeros(shape=(row, col, dim), dtype=np.int16)
    p_row, p_col, p_dim = f.__next__()
    img_x = f.__next__()

    subject_name = 'subject-%d-' % case
    label_y, _ = load(os.path.join(args.data_path, subject_name + 'label.hdr'))

    model = Unet(n_class=4)
    model.build(input_shape=(args.batch_size, 64, 64, 64, 1))
    model.load_weights(os.path.join(args.pretrained_dir, 'fold_0', 'final_weights.h5'), by_name=True)
    i, j, k = 0, 0, 0
    step = 32
    for patches in tqdm(f):
        labels = model(patches)
        labels = np.argmax(labels, axis=-1)
        b, h, w, d = labels.shape
        for label in labels:
            h1, w1, d1 = img_y[i*step: (i+1)*step, j*step: (j+1)*step, k*step: (k+1)*step].shape
            img_y[i * step: (i + 1) * step, j * step: (j + 1) * step, k * step: (k + 1) * step] = \
                label[(h - step) // 2:(h - step) // 2 + h1, (w - step) // 2:(w - step) // 2 + w1, (d - step) // 2:(d - step) // 2 + d1]
            k += 1
            if k >= p_dim:
                j += 1
                k = 0
            if j >= p_col:
                i += 1
                j, k = 0, 0

    summary_writer = tf.summary.create_file_writer(args.target_path)
    with summary_writer.as_default():
        for i in range(img_x.shape[2]):
            tf.summary.image('x', img_x[:, :, i][None, :, :, None], step=i)
            tf.summary.image('y_', img_y[:, :, i][None, :, :, None] / 4., step=i)
            tf.summary.image('y', label_y[:, :, i][None, :, :, None] / 4., step=i)
            # tf.summary.image('y_tmp', img_tmp[:, :, i][None, :, :, None] / 255., step=i)

    save(img_y, filename=os.path.join(args.target_path, subject_name + 'label.hdr'))
    img_y_read, _ = load(os.path.join(args.target_path, subject_name + 'label.hdr'))
    img_y_read = np.array(img_y_read)
    with summary_writer.as_default():
        for i in range(img_y_read.shape[2]):
            tf.summary.image('y_read', img_y_read[:, :, i][None, :, :, None] / 4., step=i)


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-path', default='/data/data/iseg2019/iSeg-2019-Training/iSeg-2019-Training/', type=str)
    parser.add_argument('-t', '--target-path', default='output-4', type=str)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    # Train Setting
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('-p', '--pretrained-dir', default='output-3/', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.target_path):
        os.mkdir(args.target_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    test_one_case(args, 1)
