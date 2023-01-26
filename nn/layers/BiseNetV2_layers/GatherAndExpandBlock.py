import tensorflow as tf
from ConvBlock import ConvBlock
# left path:
# 1 regular conv, in_chan to in_chan kernel 3
# 2 DW conv, in_chan to mid_chan kernel 3 groups in_chan
# 3 DW conv, mid_chan to mid_chan kernel 3 groups mid_chan
# 4 Separable conv, mid_chan to out_chan kernel 1 groups 1
class GatherAndExpandBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, strides=1, expansion_factor=6):
        super(GatherAndExpandBlock, self).__init__()
        bottleneck_channels = int(in_channels * expansion_factor)
        self.right_exists = False

        if strides == 2:
            self.right_exists = True
            self.left_path = tf.keras.Sequential([
                ConvBlock(in_channels, kernel_size=3),
                ConvBlock(bottleneck_channels, kernel_size=3, strides=2, groups=in_channels, relu=False),
                ConvBlock(bottleneck_channels, kernel_size=3, groups=bottleneck_channels, relu=False),
                ConvBlock(out_channels, kernel_size=1, relu=False)
            ])

            self.right_path = tf.keras.Sequential([
                ConvBlock(bottleneck_channels, kernel_size=3, strides=2, groups=in_channels, relu=False),
                ConvBlock(out_channels, kernel_size=1, relu=False)
            ])


        else:
            self.left_path = tf.keras.Sequential([
                ConvBlock(bottleneck_channels, kernel_size=3),
                ConvBlock(bottleneck_channels, kernel_size=3, groups=bottleneck_channels, relu=False),
                ConvBlock(out_channels, kernel_size=1, relu=False)
            ])

    def call(self, inputs):
        output = self.left_path(inputs)
        if self.right_exists:
            right = self.right_path(inputs)
            output = tf.keras.layers.Add()([output, right])

        output = tf.keras.layers.ReLU()(output)

        return output