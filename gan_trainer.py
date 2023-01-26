import tensorflow as tf
from nn.models.GAN_models import PGGAN
import numpy as np
import os
from datetime import date
from tqdm import tqdm
from utils.indian_dr_dataset_prep import prep_batch, load_batch, visualise_mask
import cv2
from utils.utils import log2



class GanTrainerConfig:
    # optimization parameters
    max_epochs = 10
    #batch size is used here to calculate the total ammount of data we can handle, so we only provide batch size for the smallest resolution
    batch_size = 64
    learning_rate = 1e-3
    start_res = (64,64)
    #we have a current_res thing in case we are resuming training at a specific resolution
    current_res = (64,64)
    target_res = (256,256)
    gen_ckpt_path = None
    dis_ckpt_path = None
    num_passes = 200 #number of times per epoch that the dataset is cycled through
    save_dir = None
    n_classes = 6
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
            self.config.target_res[0])
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
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                                  reduction=tf.keras.losses.Reduction.NONE)

            if self.config.gen_ckpt_path:
                _ = self.model.gen_model(np.zeros((0,self.config.img_size[0],self.config.img_size[1],3)))
                _ = self.model.dis_model(np.zeros((0, self.config.img_size[0], self.config.img_size[1], self.config.n_classes)))
                self.model.gen_model.load_weights(self.config.gen_ckpt_path)
                self.model.dis_model.load_weights(self.config.dis_ckpt_path)
                print('loaded weights')

    def generator_loss(self, disc_generated_output, gen_output, target):
        #resizing and reshaping the gan loss so that it's the same size as the l1 loss
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        gan_loss = tf.expand_dims(gan_loss, -1)
        gan_loss = tf.image.resize(gan_loss,[self.model.current_res, self.model.current_res])
        gan_loss = tf.squeeze(gan_loss)
        l1_loss = self.cce(target, gen_output)
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

        #curr_batch_size = int((self.config.batch_size*(self.config.start_res[0]**2))/(self.model.current_res**2))
        curr_batch_size = 8

        def train_step(inputs):

            def step_fn(inputs):
                #These will come in at maximum resolution
                #print('aaaaaaaaaaa ',[curr_batch_size,self.model.current_res,self.model.current_res,3])
                X, Y = inputs
                Y = tf.image.resize(Y, [self.model.current_res,self.model.current_res])
                X_current_res = tf.image.resize(X, [self.model.current_res,self.model.current_res])

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
                                                     self.model.gen_model.trainable_variables))
                self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                                     self.model.dis_model.trainable_variables))

                return gen_loss

            per_example_losses = self.strategy.run(step_fn, args=(inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        def test_step(inputs):

            def step_fn(inputs):
                X, Y = inputs
                Y = tf.image.resize(Y, [self.model.current_res, self.model.current_res])
                X_current_res = tf.image.resize(X, [self.model.current_res, self.model.current_res])
                """setting training to false to disable the dropout layers"""
                gen_output = self.model.gen_model(X, training=False)
                disc_real_output = self.model.dis_model([X_current_res, Y], training=False)
                disc_generated_output = self.model.dis_model([X_current_res, gen_output], training=False)

                gen_loss = self.generator_loss(disc_generated_output, gen_output, Y)
                dis_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
                test_iou_metric.update_state(tf.argmax(Y, axis=-1), tf.argmax(gen_output, axis=-1))
                return gen_loss

            per_example_losses = self.strategy.run(step_fn, args=(inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        with self.strategy.scope():
            for current_res_log2 in range(log2(self.config.current_res[0]), log2(self.config.target_res[0])+1):
                if self.model.current_res != 2**current_res_log2:
                    self.model.grow_model(2**current_res_log2)
                #curr_batch_size = int(
                #    (self.config.batch_size * (self.config.start_res[0] ** 2)) / (self.model.current_res**2))
                curr_batch_size = self.config.batch_size
                for epoch in range(self.config.max_epochs):
                    pbar = tqdm(range(0, (len(self.train_dataset[0]) * self.config.num_passes), self.config.batch_size))
                    for i in pbar:
                        # for i in progress_bar(range(0,(len(self.train_dataset[0])*200), self.config.batch_size), total=len(self.train_dataset[0])*200//self.config.batch_size, parent=epoch_bar):
                        inputs = load_batch(self.train_dataset[0], self.train_dataset[1],
                                            i % (len(self.train_dataset[0])), curr_batch_size,
                                            self.config.target_res, augment=True)
                        loss = train_step(inputs)
                        self.tokens += tf.reduce_sum(tf.cast(inputs[1] >= 0, tf.int32)).numpy()
                        # epoch_bar.child.comment = f'training loss : {train_loss_metric.result()} training iou : {train_iou_metric.result()}'
                        pbar.set_description(
                            f'Resolution={2**current_res_log2}, Epoch={epoch}, Gen_Loss={gen_loss_metric.result()}, '
                            f'Dis_Loss={dis_loss_metric.result()}, Train_IoU={train_iou_metric.result()}')
                    print(
                        f"resolution {2**current_res_log2} epoch {epoch + 1}: gen loss {gen_loss_metric.result():.5f}. dis loss {dis_loss_metric.result():.5f}. train iou {train_iou_metric.result():.5f}")
                    trainIoU = train_iou_metric.result()
                    gen_loss_metric.reset_states()
                    dis_loss_metric.reset_states()
                    train_iou_metric.reset_states()

                    if self.test_dataset:
                        pbar = tqdm(range(0, (len(self.test_dataset[0])), self.config.batch_size))
                        for i in pbar:
                            # for i in progress_bar(range(0, len(self.test_dataset[0]), self.config.batch_size), total=len(self.test_dataset[0])//self.config.batch_size, parent=epoch_bar):
                            inputs = load_batch(self.test_dataset[0], self.test_dataset[1],
                                                i % len(self.test_dataset[0]), curr_batch_size,
                                                self.config.target_res, augment=False)
                            loss = test_step(inputs)
                            # epoch_bar.child.comment = f'testing loss : {test_loss_metric.result()}'
                            pbar.set_description(
                                f'Epoch={epoch}, Test_Loss={test_iou_metric.result()}')
                        print(
                            f"resolution {2**current_res_log2} epoch {epoch + 1}: test iou {test_iou_metric.result():.5f}")
                        testIoU = test_iou_metric.result()
                        test_iou_metric.reset_states()

                        vis_batch = load_batch(self.test_dataset[0], self.test_dataset[1], 0, 10, self.config.target_res,
                                               augment=False)
                        if epoch % 3 == 0:
                            for i in range(10):
                                logits = self.model.gen_model(np.expand_dims(vis_batch[0][i], 0))
                                print('logits shape ',logits.shape)
                                vis = visualise_mask(tf.squeeze(logits), cv2.resize(vis_batch[0][i], (self.model.current_res,self.model.current_res)))
                                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                                vis = cv2.resize(vis, (512,512))
                                cv2.imwrite(f'{self.img_dir}resolution_{2**current_res_log2}epoch_{epoch}_img{i}.jpg', vis)
                                vis = visualise_mask(tf.squeeze(logits), np.zeros((self.model.current_res, self.model.current_res, 3)))
                                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                                vis = cv2.resize(vis, (512, 512))
                                cv2.imwrite(f'{self.img_dir}resolution_{2**current_res_log2}epoch_{epoch}_img_jm{i}.jpg', vis)
                            self.model.dis_model.save_weights(
                                f'{self.model_dir}dis_resolution{2**current_res_log2}epoch_{epoch}_trainIoU_{trainIoU:.3f}_testIoU_{testIoU:.3f}.h5')
                            self.model.gen_model.save_weights(
                                f'{self.model_dir}gen_resolution{2 ** current_res_log2}epoch_{epoch}_trainIoU_{trainIoU:.3f}_testIoU_{testIoU:.3f}.h5')
