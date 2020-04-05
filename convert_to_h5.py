import os

import numpy as np
import h5py
from medpy.io import load
import argparse
from tqdm import tqdm

# 每个case大小144*192*256，以32为步长，切成64*64*64大小的patch，每个case有3*4*6个patch
def parse_args():
    parser = argparse.ArgumentParser(description='convert_to_h5')
    # config
    parser.add_argument('-d', '--data-file-path', default='/data/data/iseg2019/iSeg-2019-Training/iSeg-2019-Training', type=str)
    parser.add_argument('-l', '--target-path', default='/data/data/iseg2019/training_h5', type=str)
    # Train Setting
    parser.add_argument('--step-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=64)

    args = parser.parse_args()

    print('data file path =', args.data_file_path)
    print('target path = ', args.target_path)
    print('step size = ', args.step_size)
    print('patch size = ', args.patch_size)

    return args


def generate_h5(args):
    if not os.path.isdir(args.target_path):
        os.mkdir(args.target_path)
    print('-------------------')
    print('Generating h5 file...')
    f = h5py.File(os.path.join(args.target_path, 'iseg2019_training.h5'), 'w')
    T1, T2, label = [], [], []
    patch_per_case = 0
    for i in tqdm(range(1, 11, 1), ascii=True, position=0, leave=True):
        subject_name = 'subject-%d-' % i
        f_T1 = os.path.join(args.data_file_path, subject_name + 'T1.hdr')
        f_T2 = os.path.join(args.data_file_path, subject_name + 'T2.hdr')
        f_label = os.path.join(args.data_file_path, subject_name + 'label.hdr')
        img_T1, header_T1 = load(f_T1)
        img_T2, header_T2 = load(f_T2)
        img_label, header_label = load(f_label)
        patch_per_case = ((img_label.shape[0] - args.patch_size + 1) // args.step_size + 1) * \
                         ((img_label.shape[1] - args.patch_size + 1) // args.step_size + 1) * \
                         ((img_label.shape[2] - args.patch_size + 1) // args.step_size + 1)
        for idx_x in range(0, img_label.shape[0] - args.patch_size + 1, args.step_size):
            for idx_y in range(0, img_label.shape[1] - args.patch_size + 1, args.step_size):
                for idx_z in range(0, img_label.shape[2] - args.patch_size + 1, args.step_size):
                    T1.append(img_T1[idx_x:idx_x + args.patch_size, idx_y:idx_y + args.patch_size, idx_z:idx_z + args.patch_size])
                    T2.append(img_T2[idx_x:idx_x + args.patch_size, idx_y:idx_y + args.patch_size, idx_z:idx_z + args.patch_size])
                    label.append(img_label[idx_x:idx_x + args.patch_size, idx_y:idx_y + args.patch_size, idx_z:idx_z + args.patch_size])
    T1 = np.array(T1)
    T2 = np.array(T2)
    label = np.array(label)
    patch_per_case = np.array([patch_per_case])
    f.create_dataset(name='T1', data=T1)
    f.create_dataset(name='T2', data=T2)
    f.create_dataset(name='label', data=label)
    f.create_dataset(name='patch_per_case', data=patch_per_case)
    f.close()


if __name__ == '__main__':
    args = parse_args()
    generate_h5(args=args)
    # test
    f = h5py.File(os.path.join(args.target_path, 'iseg2019_training.h5'), 'r')
    print(f['T1'].shape)
    print(f['T2'].shape)
    print(f['label'].shape)
    print(f['patch_per_case'][0])

