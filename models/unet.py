import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv3D, UpSampling3D, BatchNormalization, MaxPooling3D, Concatenate,\
    ReLU, Softmax, Dropout
from tensorflow.keras import Model, Sequential

import os


class Down(Model):
    def __init__(self, out_ch):
        super(Down, self).__init__()
        self.downsampling = MaxPooling3D()
        self.conv = Sequential([
            Conv3D(filters=out_ch, kernel_size=3, padding='same'),
            BatchNormalization(center=True, scale=True),
            ReLU(),
            Conv3D(filters=out_ch, kernel_size=3, padding='same'),
            BatchNormalization(center=True, scale=True),
            ReLU()
        ])
    def call(self, x):
        x = self.downsampling(x)
        x = self.conv(x)
        return x


class Up(Model):
    def __init__(self, out_ch):
        super(Up, self).__init__()
        self.upsampling = UpSampling3D()
        self.concat = Concatenate()
        self.conv = Sequential([
            Conv3D(filters=out_ch, kernel_size=3, padding='same'),
            BatchNormalization(center=True, scale=True),
            ReLU(),
            Conv3D(filters=out_ch, kernel_size=3, padding='same'),
            BatchNormalization(center=True, scale=True),
            ReLU()
        ])

    def call(self, x):
        x1, x2 = x[0], x[1]
        x = self.upsampling(x1)
        x = self.concat([x, x2])
        x = self.conv(x)
        return x




class Unet(Model):
    def __init__(self, n_class):
        super(Unet, self).__init__()
        self.pre_conv = Sequential([
            Conv3D(filters=32, kernel_size=3, padding='same'),
            BatchNormalization(center=True, scale=True),
            ReLU(),
            Conv3D(filters=32, kernel_size=3, padding='same'),
            BatchNormalization(center=True, scale=True),
            ReLU()
        ])
        self.down1 = Down(out_ch=64)
        self.down2 = Down(out_ch=128)
        self.down3 = Down(out_ch=256)
        self.down4 = Down(out_ch=512)
        self.dropout = Dropout(0.2)
        self.up4 = Up(out_ch=256)
        self.up3 = Up(out_ch=128)
        self.up2 = Up(out_ch=64)
        self.up1 = Up(out_ch=32)
        self.conv_out = Sequential([
            Conv3D(filters=n_class, kernel_size=3, padding='same'),
            Softmax()
        ])
    def call(self, x):
        skip1 = self.pre_conv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        x = self.dropout(x)
        x = self.up4([x, skip4])
        x = self.up3([x, skip3])
        x = self.up2([x, skip2])
        x = self.up1([x, skip1])
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = Unet(n_class=4)
    model.build(input_shape=(1, 64, 64, 64, 1))
    model.summary()

    for layer in model.layers:
        for weight in layer.weights:
            print(weight.name, weight.shape)
