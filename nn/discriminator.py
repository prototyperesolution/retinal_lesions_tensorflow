"""discriminator for GAN architecture"""

import tensorflow as tf
from nn.gan_layers import build_conv_block, build_depthwise_block, build_transpose_block
from utils.utils import log2

"""all necessary imports"""
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import layers



class Discriminator:
    def __init__(self, start_res, target_res, n_classes):
        self.n_classes = n_classes
        self.start_res_log = log2(start_res)
        self.target_res_log = log2(target_res)

        # filter size to use at each stage, keys are log2(resolution)
        self.filter_nums = {
            0: 512,
            1: 512,
            2: 512,  # 4x4
            3: 512,  # 8x8
            4: 512,  # 16x16
            5: 512,  # 32x32
            6: 256,  # 64x64
            7: 128,  # 128x128
            8: 64,  # 256x256
            9: 32,  # 512x512
            10: 16,
        }  # 1024x1024

        self.discriminator_blocks = []
        self.input_blocks = []
        self.discriminator_depth = self.target_res_log - self.start_res_log

        for i in range(self.start_res_log, self.target_res_log + 1):
            discriminator_input_shape = (2 ** i, (2 ** i), self.filter_nums[i])
            self.discriminator_blocks.append(self.build_discriminator_block(discriminator_input_shape,
                                                                            self.filter_nums[i - 1],
                                                                            2 ** i))
            self.input_blocks.append(self.build_input_block(self.filter_nums[i], 2 ** i))

    # the from_input block will take two inputs, those being the image input and the class input (mask)
    def build_input_block(self, n_filters, res):
        # n_channels is output channels
        # input is 1 channel as it's just a greyscale image
        init = RandomNormal(stddev=0.02)
        image_input = tf.keras.layers.Input(shape=(res, res, 3), name=f"imgin_{res}")
        mask_input = tf.keras.layers.Input(shape=(res, res, self.n_classes), name=f"maskin_{res}")

        x = Concatenate()([image_input, mask_input])
        # x = tf.keras.layers.GaussianNoise(stddev = 0.15)(x)
        x = Conv2D(n_filters, (3, 3), strides=(1), kernel_initializer=init, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        return keras.Model([image_input, mask_input], x, name=f"inblock_{res}")

    def build_discriminator_block(self, input_shape, n_channels, res, stride=2):
        # the inputs for discriminator block will not be two inputs as that is handled by input block
        init = RandomNormal(stddev=0.02)
        input_tensor = tf.keras.layers.Input(shape=(input_shape))
        x = Conv2D(n_channels, (3, 3), strides=stride, kernel_initializer=init, padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return keras.Model(input_tensor, x, name=f"disc_{res}")

    def grow(self, res):
        # similar to the generator grow function but simpler
        init = RandomNormal(stddev=0.02)
        res_log = log2(res) - self.start_res_log
        stages_current = res_log
        input_image_tensor = layers.Input(shape=(res, res, 3))
        input_mask_tensor = layers.Input(shape=(res, res, self.n_classes))
        x = self.input_blocks[stages_current]([input_image_tensor, input_mask_tensor])
        # weird looking for loop cos we're counting backwards through the discriminator blocks
        for i in range(stages_current, -1, -1):
            x = self.discriminator_blocks[i](x)
        # final layer of disc
        x = Conv2D(1, (3, 3), (1, 1), padding='same', kernel_initializer=init)(x)
        # x = tf.keras.layers.Activation('sigmoid')(x)
        return keras.Model([input_image_tensor, input_mask_tensor], x)

