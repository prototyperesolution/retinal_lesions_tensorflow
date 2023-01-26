import tensorflow as tf

class ConvBlock(tf.keras.Model):
    def __init__(self, channels,
                 kernel_size=3,
                 strides=1,
                 groups=1,
                 bias=False,
                 relu=False):
        self.relu_needed = relu
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_size,
                                           strides=strides, padding='same',
                                           groups=groups, use_bias=bias)
        self.BN = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()

    def call(self, inputs):
        conved = self.conv(inputs)
        normed = self.BN(conved)
        if self.relu_needed:
            relued = self.ReLU(normed)
            return relued
        else:
            return normed