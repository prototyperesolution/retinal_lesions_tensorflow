import tensorflow as tf


class DeepLabV3_Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, n_classes, out_size, batchnorm=True):
        super().__init__()
        self.batchnorm = []
        if batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters, 1, (1, 1))
        self.conv2 = tf.keras.layers.Conv2D(n_classes, 3, (1, 1), "SAME")
        self.out_size = out_size

    def call(self, x1, x2):
        x2 = self.conv1(x2)
        x2 = self.batchnorm(x2)
        x2 = tf.nn.relu(x2)
        x2 = tf.image.resize(images=x2, size=x1.shape[1:-1])
        x = tf.concat([x1, x2], axis=-1)
        x = self.conv2(x)
        x = tf.image.resize(images=x, size=self.out_size)
        return x

    """testing"""