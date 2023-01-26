import tensorflow as tf
from ConvBlock import ConvBlock

class ContextEmbeddingBlock(tf.keras.Model):
    def __init__(self, in_channels):
        super(ContextEmbeddingBlock, self).__init__()
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.BatchNormalization(),
            ConvBlock(channels=in_channels, strides=1, kernel_size=1)
        ])
        self.last_conv = ConvBlock(channels=in_channels, strides=1, kernel_size=3, relu=False)

    def call(self, inputs):
        main = self.main_path(inputs)
        added = main + inputs
        output = self.last_conv(added)

        return output