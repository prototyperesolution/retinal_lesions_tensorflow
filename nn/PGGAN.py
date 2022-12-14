"""full GAN architecture"""

import tensorflow as tf
from nn.gan_layers import build_conv_block, build_depthwise_block, build_transpose_block
from nn.generator import Generator
from nn.discriminator import Discriminator
from utils import log2

"""all necessary imports"""
from tensorflow import keras

LAMBDA = 100


class Prog_Seg_GAN(tf.keras.Model):
    def __init__(self, start_res=16, target_res=256, n_classes=21):
        super(Prog_Seg_GAN, self).__init__()
        self.n_classes = n_classes
        self.g_builder = Generator(2 ** (log2(start_res) - 1), target_res, n_classes)
        self.d_builder = Discriminator(start_res, target_res, n_classes)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.current_res = target_res

    def grow_model(self, res):
        self.gen_model = self.g_builder.grow(res)
        self.dis_model = self.d_builder.grow(res)
        self.current_res = res

        print(f"\nModel resolution:{res}x{res * 2}")

    def compile(self, steps_per_epoch, res, d_optimizer, g_optimizer, *args, **kwargs):
        self.steps_per_epoch = steps_per_epoch
        if res != self.current_res:
            self.grow_model(res)

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.d_real_loss_metric = keras.metrics.Mean(name='d_real_loss')
        self.d_gen_loss_metric = keras.metrics.Mean(name='d_gen_loss')
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.l1_loss_metric = keras.metrics.Mean(name="g_l1_loss")
        self.gen_gan_loss_metric = keras.metrics.Mean(name="g_gan_loss")
        super(Prog_Seg_GAN, self).compile(*args, **kwargs)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.nn.softmax_cross_entropy_with_logits(target, gen_output, axis=-1, name=None)

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def one_hot_encode(self, mask):
        maxes = tf.math.argmax(mask, axis=-1)
        return (tf.one_hot(maxes, np.shape(mask)[-1]))

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss, real_loss, generated_loss

    def train_step(self, inputs):
        # inputs contains images and masks
        # input_image_target res is input to generator, input_image_current_res is input to disc
        input_image_target_res, target = inputs
        input_image_current_res = tf.image.resize(
            input_image_target_res, (self.current_res, self.current_res * 2),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.gen_model(input_image_target_res, training=True)

            gen_output_one_hot = self.one_hot_encode(gen_output)

            disc_real_output = self.dis_model([input_image_current_res, target], training=True)
            disc_generated_output = self.dis_model([input_image_current_res, gen_output_one_hot], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss, disc_r_loss, disc_g_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.gen_model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.dis_model.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_gradients,
                                             self.gen_model.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                             self.dis_model.trainable_variables))

        self.d_loss_metric.update_state(disc_loss)
        self.g_loss_metric.update_state(gen_total_loss)
        self.l1_loss_metric.update_state(gen_l1_loss)
        self.d_real_loss_metric.update_state(disc_r_loss)
        self.d_gen_loss_metric.update_state(disc_g_loss)
        self.gen_gan_loss_metric.update_state(gen_gan_loss)

        return {
            'G_total_loss': self.g_loss_metric.result(),
            'G_l1_loss': self.l1_loss_metric.result(),
            'G_gan_loss': self.gen_gan_loss_metric.result(),
            'D_loss': self.d_loss_metric.result(),
            'D_real_loss': self.d_real_loss_metric.result(),
            'D_gen_loss': self.d_gen_loss_metric.result()
        }