import tensorflow as tf
from ConvBlock import ConvBlock
class DetailBranch(tf.keras.Model):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.s1 = tf.keras.Sequential([
            ConvBlock(channels=64, kernel_size=3, strides=2),
            ConvBlock(channels=64, kernel_size=3, strides=1)
        ])
        self.s2 = tf.keras.Sequential([
            ConvBlock(channels=64, kernel_size=3, strides=2),
            ConvBlock(channels=64, kernel_size=3, strides=1),
            ConvBlock(channels=64, kernel_size=3, strides=1)
        ])
        self.s3 = tf.keras.Sequential([
            ConvBlock(channels=128, kernel_size=3, strides=2),
            ConvBlock(channels=128, kernel_size=3, strides=1),
            ConvBlock(channels=128, kernel_size=3, strides=1)
        ])

    def call(self, inputs):
        s1_result = self.s1(inputs)
        s2_result = self.s2(s1_result)
        s3_result = self.s3(s2_result)

        return s3_result