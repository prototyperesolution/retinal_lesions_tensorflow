import tensorflow as tf
from nn import PGGAN
import numpy as np
import os
from datetime import date

def log2(num):
    return int(np.log2(num))

class GanTrainerConfig:
    # optimization parameters
    max_epochs = 10
    #batch size is used here to calculate the total ammount of data we can handle, so we only provide batch size for the smallest resolution
    batch_size = 64
    learning_rate = 1e-3
    start_res = (64,64)
    target_res = (256,256)
    gen_ckpt_path = None
    dis_ckpt_path = None
    num_passes = 200 #number of times per epoch that the dataset is cycled through
    save_dir = None
    n_classes = 3
    LAMBDA = 100

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GanTrainer:

    def __init__(self, model, model_config, train_dataset, test_dataset, config):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=len(self.train_dataset[1][0]))
        self.config = config
        self.tokens = 0
        self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        self.savedir = self.config.save_dir + str(date.today()) + '_' + str(self.config.batch_size) + '_' + str(
            self.config.img_size[0])
        self.model_dir = self.savedir + '/models/'
        self.img_dir = self.savedir + '/image results/'
        if os.path.isdir(self.savedir) == False:
            os.mkdir(self.savedir)
        if os.path.isdir(self.model_dir) == False:
            os.mkdir(self.model_dir)
        if os.path.isdir(self.img_dir) == False:
            os.mkdir(self.img_dir)

        with self.strategy.scope():
            self.model = model(**model_config)
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                               reduction=tf.keras.losses.Reduction.NONE)
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)

            if self.config.ckpt_path:
                _ = self.model.gen_model(np.zeros((0,self.config.img_size[0],self.config.img_size[1],3)))
                _ = self.model.dis_model(np.zeros((0, self.config.img_size[0], self.config.img_size[1], self.config.n_classes)))
                self.model.gen_model.load_weights(self.config.gen_ckpt_path)
                self.model.dis_model.load_weights(self.config.dis_ckpt_path)
                print('loaded weights')

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = self.cce(target, gen_output, axis=-1, name=None)

        total_gen_loss = gan_loss + (self.config.LAMBDA * l1_loss)
        return total_gen_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def train(self):

        gen_loss_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        dis_loss_metric = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
        train_iou_metric = tf.keras.metrics.MeanIoU(num_classes=self.config.n_classes)  # , sparse_y_true=False, sparse_y_pred=False)
        test_iou_metric = tf.keras.metrics.MeanIoU(num_classes=self.config.n_classes)  # sparse_y_true=False, sparse_y_pred=False)

        curr_batch_size = (self.config.batch_size*(self.config.start_res[0]**2))/(self.model.current_res)

        def train_step(inputs):

            def step_fn(inputs):
                #These will come in at maximum resolution
                X, Y = inputs
                Y = tf.image.resize(Y, (curr_batch_size,self.model.current_res,self.model.current_res,3))
                X_current_res = tf.image.resize(X, (curr_batch_size,self.model.current_res,self.model.current_res,3))

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_output = self.model.gen_model(X, training=True)
                    disc_real_output = self.model.dis_model([X_current_res, Y], training=True)
                    disc_generated_output = self.model.dis_model([X_current_res, gen_output], training=True)

                    gen_loss = self.generator_loss(disc_generated_output, gen_output, Y)
                    dis_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

                    dis_loss_metric.update_state(dis_loss)
                    gen_loss_metric.update_state(gen_loss)
                    train_iou_metric.update_state(tf.argmax(Y, axis=-1), tf.argmax(gen_output, axis=-1))

                generator_gradients = gen_tape.gradient(gen_loss,
                                                        self.model.gen_model.trainable_variables)
                discriminator_gradients = disc_tape.gradient(dis_loss,
                                                             self.model.dis_model.trainable_variables)

                self.g_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.gen_model.trainable_variables))
                self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                                     self.dis_model.trainable_variables))

                return gen_loss+dis_loss

            per_example_losses = self.strategy.run(step_fn, args=(inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        def test_step(inputs):

            def step_fn(inputs):
                X, Y = inputs
                Y = tf.image.resize(Y, (curr_batch_size, self.model.current_res, self.model.current_res, 3))
                X_current_res = tf.image.resize(X, (curr_batch_size, self.model.current_res, self.model.current_res, 3))
                """setting training to false to disable the dropout layers"""
                gen_output = self.model.gen_model(X, training=False)
                disc_real_output = self.model.dis_model([X_current_res, Y], training=False)
                disc_generated_output = self.model.dis_model([X_current_res, gen_output], training=False)

                gen_loss = self.generator_loss(disc_generated_output, gen_output, Y)
                dis_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
                test_iou_metric.update_state(tf.argmax(Y, axis=-1), tf.argmax(gen_output, axis=-1))
                return gen_loss+dis_loss

            per_example_losses = self.strategy.run(step_fn, args=(inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss
