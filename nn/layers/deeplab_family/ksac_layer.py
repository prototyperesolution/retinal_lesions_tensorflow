"""kernel sharing atrous convolutional layer"""
import tensorflow as tf
import numpy as np
import cv2


class KSAC_layer(tf.keras.layers.Layer):
    def __init__(self, input_shape, filters, dilation_rates=[6, 12, 18], batchnorm=True):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.batchnorms = []
        self.filters = filters
        if batchnorm:
            self.batchnorms = [tf.keras.layers.BatchNormalization() for _ in dilation_rates]
        self.kernel_initializer = tf.keras.initializers.GlorotUniform()
        self.kernel_shape = (3, 3, input_shape[-1], filters)
        self.kernel = tf.Variable(self.kernel_initializer(self.kernel_shape), trainable=True)

    def call(self, x, training=False):
        feature_maps = [tf.nn.conv2d(x, self.kernel, (1, 1), 'SAME', dilations=d) for d in self.dilation_rates]
        if len(self.batchnorms) > 0:
            for i in range(len(feature_maps)):
                feature_maps[i] = self.batchnorms[i](feature_maps[i], training=training)
        return sum(feature_maps)

