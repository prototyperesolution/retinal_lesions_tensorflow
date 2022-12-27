import cv2
import tensorflow as tf
import numpy as np
from fastprogress import master_bar, progress_bar
from utils.indian_dr_dataset_prep import prep_batch, load_batch, visualise_mask
from utils.losses import focal_loss
import random
import math
from tqdm import tqdm
import cv2
from datetime import date
import os

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    img_size = (256,256)
    ckpt_path = None
    num_passes = 200 #number of times per epoch that the dataset is cycled through
    save_dir = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    """handles the training of the model"""

    def __init__(self, model, model_config, train_dataset, test_dataset, config):
        """datasets are stored as x,y arrays of filenames"""
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iou_metric = tf.keras.metrics.MeanIoU( num_classes =len(self.train_dataset[1][0]))
        self.config = config
        self.tokens = 0
        self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        self.savedir = self.config.save_dir + str(date.today()) +'_'+ str(self.config.batch_size) +'_'+ str(self.config.img_size[0])
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
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                               reduction=tf.keras.losses.Reduction.NONE)

            if self.config.ckpt_path:
                _ = self.model(np.zeros((0,self.config.img_size[0],self.config.img_size[1],3)))
                self.model.load_weights(self.config.ckpt_path)
                print('loaded weights')

    def train(self):

        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        test_loss_metric = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)
        train_iou_metric = tf.keras.metrics.MeanIoU(num_classes=6)#, sparse_y_true=False, sparse_y_pred=False)
        test_iou_metric = tf.keras.metrics.MeanIoU(num_classes=6)# sparse_y_true=False, sparse_y_pred=False)


        def train_step(inputs):

            def step_fn(inputs):
                X, Y = inputs

                with tf.GradientTape() as tape:
                    logits = self.model(X, training=True)
                    l1_loss = self.cce(Y, logits)

                    train_iou_metric.update_state(tf.argmax(Y, axis=-1), tf.argmax(logits, axis=-1))

                grads = tape.gradient(l1_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
                return l1_loss

            per_example_losses = self.strategy.run(step_fn, args=(inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        def test_step(inputs):

            def step_fn(inputs):
                X, Y = inputs
                """setting training to false to disable the dropout layers"""
                logits = self.model(X, training=False)
                l1_loss = self.cce(Y, logits)
                test_iou_metric.update_state(tf.argmax(Y, axis=-1), tf.argmax(logits, axis=-1))
                return l1_loss

            per_example_losses = self.strategy.run(step_fn, args=(inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        #epoch_bar = master_bar(range(self.config.max_epochs))
        with self.strategy.scope():
            for epoch in range(self.config.max_epochs):
                pbar = tqdm(range(0,(len(self.train_dataset[0])*self.config.num_passes), self.config.batch_size))
                for i in pbar:
                #for i in progress_bar(range(0,(len(self.train_dataset[0])*200), self.config.batch_size), total=len(self.train_dataset[0])*200//self.config.batch_size, parent=epoch_bar):
                    inputs = load_batch(self.train_dataset[0], self.train_dataset[1], i%(len(self.train_dataset[0])), self.config.batch_size,self.config.img_size,augment=True)
                    loss = train_step(inputs)
                    self.tokens += tf.reduce_sum(tf.cast(inputs[1] >= 0, tf.int32)).numpy()
                    train_loss_metric(loss)
                    #epoch_bar.child.comment = f'training loss : {train_loss_metric.result()} training iou : {train_iou_metric.result()}'
                    pbar.set_description(
                        f'Epoch={epoch}, Train_Loss={train_loss_metric.result()}, Train_IoU={train_iou_metric.result()}')
                print(
                    f"epoch {epoch + 1}: train loss {train_loss_metric.result():.5f}. train iou {train_iou_metric.result():.5f}")
                trainIoU = train_iou_metric.result()
                train_loss_metric.reset_states()
                train_iou_metric.reset_states()

                if self.test_dataset:
                    pbar = tqdm(range(0, (len(self.test_dataset[0])), self.config.batch_size))
                    for i in pbar:
                    #for i in progress_bar(range(0, len(self.test_dataset[0]), self.config.batch_size), total=len(self.test_dataset[0])//self.config.batch_size, parent=epoch_bar):
                        inputs = load_batch(self.test_dataset[0], self.test_dataset[1], i%len(self.test_dataset[0]), self.config.batch_size,self.config.img_size,augment=False)
                        loss = test_step(inputs)
                        test_loss_metric(loss)
                        #epoch_bar.child.comment = f'testing loss : {test_loss_metric.result()}'
                        pbar.set_description(
                            f'Epoch={epoch}, Test_Loss={test_loss_metric.result()}')
                    print(
                        f"epoch {epoch + 1}: test loss {test_loss_metric.result():.5f}. test iou {test_iou_metric.result():.5f}")
                    testIoU = test_iou_metric.result()
                    test_loss_metric.reset_states()
                    test_iou_metric.reset_states()

                    vis_batch = load_batch(self.test_dataset[0], self.test_dataset[1], 0, 10, self.config.img_size, augment=False)
                    if epoch % 15 == 0:
                        for i in range(10):
                            logits = self.model(np.expand_dims(vis_batch[0][i],0))
                            vis = visualise_mask(tf.squeeze(logits), vis_batch[0][i])
                            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(f'{self.img_dir}epoch_{epoch}_img{i}.jpg', vis)
                            vis = visualise_mask(tf.squeeze(logits), np.zeros(np.shape(vis)))
                            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(f'{self.img_dir}epoch_{epoch}_img_jm{i}.jpg', vis)
                        self.model.save_weights(f'{self.model_dir}epoch_{epoch}_trainIoU_{trainIoU:.3f}_testIoU_{testIoU:.3f}.h5')

