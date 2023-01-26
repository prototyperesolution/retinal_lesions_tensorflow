import tensorflow as tf
from ConvBlock import ConvBlock

class BilateralGuidedAggregation(tf.keras.layers.Layer):
    def __init__(self):
        super(BilateralGuidedAggregation, self).__init__()

        self.detail_left = tf.keras.Sequential([
            ConvBlock(128, kernel_size=3, groups=128),
            tf.keras.layers.Conv2D(filters=128, kernel_size=1, groups=1, padding='same')
        ])
        self.detail_right = tf.keras.Sequential([
            ConvBlock(128, kernel_size=3, strides=2),
            tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')
        ])

        self.semantic_left = tf.keras.Sequential([
            ConvBlock(128, kernel_size=3),
            tf.keras.layers.UpSampling2D(size=(4, 4)),
            # tf.keras.activations.sigmoid()
        ])

        self.semantic_right = tf.keras.Sequential([
            ConvBlock(128, kernel_size=3, groups=128),
            ConvBlock(128, kernel_size=1)
            # tf.keras.activations.sigmoid()
        ])

        self.final_conv = tf.keras.Sequential([
            ConvBlock(128, kernel_size=3)
        ])

    def call(self, inputs):
        details, semantics = inputs
        dsize = details.shape[1:3]

        # leftmost output
        left_sem = tf.keras.activations.sigmoid(self.semantic_left(semantics))
        left_det = self.detail_left(details)
        left = left_sem * left_det

        # rightmost output
        right_sem = tf.keras.activations.sigmoid(self.semantic_right(semantics))
        right_det = self.detail_right(details)
        right = right_sem * right_det
        right = tf.image.resize(right, dsize)

        total = tf.keras.layers.Add()([right, left])
        output = self.final_conv(total)

        return output