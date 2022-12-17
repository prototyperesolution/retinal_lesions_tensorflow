import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
from nn.layers.ksac_block import KSAC_block
import numpy as np
from nn.layers.DeepLabV3_decoder import DeepLabV3_Decoder
from utils.losses import focal_loss

class KSAC_network(tf.keras.Model):
    def __init__(self, input_shape, n_classes, filters=128, dilation_rate=[6,12,18], batchnorm=True):
        super().__init__()
        resnet_backbone = applications.resnet50.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
            classes=1000,
        )
        resnet_backbone = tf.keras.Model(inputs=resnet_backbone.inputs,
                                         outputs=[resnet_backbone.get_layer('conv3_block4_out').output,
                                                  resnet_backbone.get_layer('conv4_block6_out').output])


        x = tf.keras.Input(input_shape)
        x1, x2 = resnet_backbone(x)
        x2 = KSAC_block(filters, x2.shape, dilation_rate, batchnorm)(x2)
        logits = DeepLabV3_Decoder(filters, n_classes, input_shape[:-1])(x1,x2)
        self.model = tf.keras.Model(inputs=x, outputs=logits)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def compile(self, optimizer, *args, **kwargs):
        self.focal_loss_metric = keras.metrics.Mean(name="focal_loss")
        self.accuracy_metric = keras.metrics.Mean(name='accuracy')
        self.optimizer = optimizer
        super(KSAC_network, self).compile(*args, **kwargs)

    def train_step(self,inputs):
        images, masks = inputs
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss = tf.reduce_mean(focal_loss(masks, logits))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.focal_loss_metric.update_state(loss)

        return {
            'l1_loss': self.focal_loss_metric.result(),
            #'accuracy': self.accuracy_metric.result()
        }

test_ksac = KSAC_network(input_shape=(256,256,3), n_classes=3, filters=50)