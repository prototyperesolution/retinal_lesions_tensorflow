"""full GAN architecture"""

import tensorflow as tf
from nn.train.GAN_models.gan_layers import build_conv_block, build_depthwise_block, build_transpose_block
from nn.train.GAN_models.generator import Generator
from nn.train.GAN_models.discriminator import Discriminator
from utils.utils import log2

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

        print(f"\nModel resolution:{res}x{res}")


    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.nn.softmax_cross_entropy_with_logits(target, gen_output, axis=-1, name=None)

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss, real_loss, generated_loss
