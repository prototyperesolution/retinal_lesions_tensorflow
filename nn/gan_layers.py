"""blocks necessary for other components"""

import tensorflow as tf

def build_conv_block(input_shape, n_filters, kernel_size=[3, 3]):
    """
    Builds the conv block for MobileNets
    Apply successivly a 2D convolution, BatchNormalization relu
    """
    # Skip pointwise by setting num_outputs=Non
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.Conv2D(n_filters, kernel_size=[1, 1], padding = 'same')(input_tensor)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    return tf.keras.Model(input_tensor, net)

def build_depthwise_block(input_shape, n_filters, kernel_size=[3, 3]):
    """
    Builds the Depthwise Separable conv block for MobileNets
    Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
    """
    # Skip pointwise by setting num_outputs=None
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.SeparableConv2D(filters = input_shape[-1], depth_multiplier=1, kernel_size=[3,3], padding='same')(input_tensor)

    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Conv2D(n_filters, kernel_size=[1, 1], padding = 'same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    return tf.keras.Model(input_tensor, net)

def build_transpose_block(input_shape, n_filters, kernel_size=[3, 3]):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.Conv2DTranspose(n_filters, strides = (2,2), kernel_size=kernel_size, padding='same')(input_tensor)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    return tf.keras.Model(input_tensor, net)