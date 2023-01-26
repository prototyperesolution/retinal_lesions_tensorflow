import tensorflow as tf

class SegmentHead(tf.keras.Model):
    # this is a model rather than layer so that weights can be saved and retrieved
    def __init__(self, in_channels, n_classes, dropout=0.1, expansion_factor=6, size=None):
        self.size = size
        super(SegmentHead, self).__init__()
        bottleneck_channels = in_channels * expansion_factor
        self.main_path = tf.keras.Sequential([
            ConvBlock(channels=bottleneck_channels, kernel_size=3),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(filters=n_classes, kernel_size=1, padding='same')
        ])

    def call(self, inputs):
        features = inputs
        main = self.main_path(inputs)

        if self.size:
            upsampled = tf.image.resize(
                main,
                self.size,
                name='segmentation_head_logits'
            )
            return upsampled  # tf.keras.activations.sigmoid(upsampled)
        else:
            return main  # tf.keras.activations.sigmoid(main)