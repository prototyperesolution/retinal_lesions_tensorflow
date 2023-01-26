"""pooling layer for KSAC network, does global average pooling"""

import tensorflow as tf
import numpy as np
import cv2

class KSAC_pooling(tf.keras.layers.Layer):
    def __init__(self, filters, batchnorm = False):
        super().__init__()
        self.filters = filters
        self.batchnorm = []
        if batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
        self.conv_layer = tf.keras.layers.Conv2D(filters, 1, (1,1))

    def call(self, x):
        x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = self.conv_layer(x)
        if self.batchnorm != []:
            x = self.batchnorm(x)
        return tf.image.resize(images=x, size=x.shape[1:-1])