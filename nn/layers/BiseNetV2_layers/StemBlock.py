import tensorflow as tf
from ConvBlock import ConvBlock

class StemBlock(tf.keras.Model):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv1 = ConvBlock(channels=16, strides=2)
        self.left_path = tf.keras.Sequential([
            ConvBlock(channels=8, kernel_size=1),
            ConvBlock(channels=16, kernel_size=3, strides=2)
        ])
        self.right_path = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        self.concat_conv = ConvBlock(channels=16)

    def call(self, inputs):
        conved = self.conv1(inputs)
        left = self.left_path(conved)
        right = self.right_path(conved)

        concat = tf.concat((left, right), axis=-1)
        output = self.concat_conv(concat)
        return output