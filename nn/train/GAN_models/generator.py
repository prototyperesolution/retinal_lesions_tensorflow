"""generator for GAN architecture"""

import tensorflow as tf
from nn.train.GAN_models.gan_layers import build_conv_block, build_depthwise_block, build_transpose_block
from utils.utils import log2

"""all necessary imports"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D



class Generator(tf.keras.Model):
    def __init__(self, start_res, target_res, n_classes):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.current_res = None
        self.start_res_log = log2(start_res)
        self.target_res_log = log2(target_res)
        self.encoder_depth = (self.target_res_log - self.start_res_log)
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.output_blocks = []
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
        }

        self.input_block = self.build_input_block(self.filter_nums[self.target_res_log], 2 ** self.target_res_log)

        for i in range(0, self.encoder_depth):
            encoder_input_shape = (2 ** (self.target_res_log - i), (2 ** (self.target_res_log - i)),
                                   self.filter_nums[self.target_res_log - i])
            encoder_filter_nums = self.filter_nums[self.target_res_log - i - 1]

            self.encoder_blocks.append(self.build_encoder_block(encoder_filter_nums,
                                                                encoder_input_shape, 2 ** (self.target_res_log - i)))

        for i in range(0, self.encoder_depth):
            decoder_input_shape = (2 ** (self.start_res_log + i), (2 ** (self.start_res_log + i)),
                                   self.filter_nums[self.start_res_log + i])

            decoder_filter_nums = self.filter_nums[self.start_res_log + i + 1]

            output_block_shape = (2 ** (self.start_res_log + i + 1), (2 ** (self.start_res_log + i + 1)),
                                  decoder_filter_nums)

            self.decoder_blocks.append(self.build_decoder_block(decoder_filter_nums, decoder_input_shape,
                                                                2 ** (self.start_res_log + i)))

            self.output_blocks.append(self.build_output_block(output_block_shape,
                                                              2 ** (self.start_res_log + i)))

    def build_encoder_block(self, n_filters, input_shape, res, stride=2):
        init = RandomNormal(stddev=0.02)
        input_tensor = tf.keras.layers.Input(shape=(input_shape), name=f"enc_{res}")
        e = build_conv_block(input_shape, n_filters)(input_tensor)
        e = build_conv_block((input_shape[0], input_shape[1], n_filters), n_filters)(e)
        if log2(res) <= (self.target_res_log - 2):
            e = build_conv_block((input_shape[0], input_shape[1], n_filters), n_filters)(e)
        e = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(stride, stride))(e)
        return keras.Model(input_tensor, e, name=f"enc_{res}")

    def build_input_block(self, n_filters, res):
        init = RandomNormal(stddev=0.02)
        input_tensor = tf.keras.layers.Input(shape=(res,res, 3), name=f"input_{res}")
        i = Conv2D(n_filters, (3, 3), (1, 1), padding='same', kernel_initializer=init)(input_tensor)
        i = tf.keras.layers.LeakyReLU(0.2)(i)
        return keras.Model(input_tensor, i, name=f'input_{res}')

    def build_output_block(self, input_shape, res):
        init = RandomNormal(stddev=0.02)
        input_tensor = tf.keras.layers.Input(shape=(input_shape), name=f"out_{res}")
        o = Conv2D(self.n_classes, (3, 3), (1, 1), padding='same', kernel_initializer=init, activation='softmax')(input_tensor)
        # o = tf.keras.layers.Activation('sigmoid')(o)
        # o = tf.keras.layers.Softmax(axis=-1) (o)
        return keras.Model(input_tensor, o, name=f"out_{res}")

    def build_decoder_block(self, n_filters, input_shape, res, stride=2):
        # a special case for the decoder block is the one in the middle, it doesn't need a skip
        # connection as it directly connects to the block before it anyway
        init = RandomNormal(stddev=0.02)
        input_tensor = tf.keras.layers.Input(shape=(input_shape), name=f"dec_input_{2 ** (log2(res) - 1)}")
        d = build_transpose_block(input_shape, input_shape[-1])(input_tensor)
        d = build_conv_block((input_shape[0] * 2, input_shape[1] * 2, input_shape[-1]), input_shape[-1])(d)
        if log2(res) <= (self.target_res_log - 2):
            d = build_conv_block((input_shape[0] * 2, input_shape[1] * 2, input_shape[-1]), input_shape[-1])(d)
        d = build_conv_block((input_shape[0] * 2, input_shape[1] * 2, input_shape[-1]), n_filters)(d)
        return keras.Model(input_tensor, d, name=f"dec_{res}")

    def grow(self, res):
        input_tensor = tf.keras.layers.Input(shape=(2 ** self.target_res_log, (2 ** self.target_res_log), 3))
        idx = (log2(res) - self.start_res_log) + 1
        skip_tensors = []
        # encoding
        x = self.input_block(input_tensor)
        for block in self.encoder_blocks:
            x = block(x)
            skip_tensors.append(x)
        skip_tensors = list(reversed(skip_tensors[:-1]))
        print('_____')
        for tensor in skip_tensors:
            print(tensor.shape)
        print('_____')
        print(idx - 1)
        # decoding
        x = self.decoder_blocks[0](x)
        if idx != 1:
            x = tf.add(x, skip_tensors[0])
        if idx == 1:
            x = self.output_blocks[0](x)
        else:
            for i in range(1, idx - 1):
                x = self.decoder_blocks[i]([x])
                if i < len(skip_tensors):
                    print('a')
                    x = tf.add(x, skip_tensors[i])
            i = idx - 2
            x = self.output_blocks[i](x)
        return tf.keras.Model(input_tensor, x)

