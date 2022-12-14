"""takes as input a regular tensor then convolves it, then concatenates this with the global average pooling layer and ksac layer"""
import tensorflow as tf
from nn.layers.ksac_layer import KSAC_layer
from nn.layers.ksac_pooling_layer import KSAC_pooling


class KSAC_block(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape, dilation_rate=[6, 12, 18], batchnorm=True):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 1, (1, 1))
        self.batchnorm = []
        if batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()


        self.ksac_layer = KSAC_layer(input_shape, filters, dilation_rate, batchnorm)
        self.ksac_pooling = KSAC_pooling(filters, batchnorm)
        self.bias = tf.Variable(tf.zeros_initializer()((filters,)), trainable=True, name='bias')

    def call(self, x):
        y = self.conv1(x)
        if self.batchnorm != []:
            y = self.batchnorm(y)
        return tf.nn.relu(y + self.ksac_layer(x) + self.ksac_pooling(x) + self.bias)


