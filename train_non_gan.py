from trainer import Trainer, TrainerConfig
from nn.create_ksac_network import KSAC_network
from utils.indian_dr_dataset_prep import prep_dataset, load_dataset
import tensorflow as tf
import numpy as np

print('ver',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_SIZE = (256,256)

train_x,train_y,test_x,test_y = load_dataset('D:/indian dr dataset', IMG_SIZE)

train_dataset = (train_x,train_y)
test_dataset = (test_x,test_y)



tconf = TrainerConfig(max_epochs=100, batch_size=32, learning_rate=1e-4, batches_per_epoch=10, img_size = IMG_SIZE)
mconf = {'input_shape':(IMG_SIZE[0],IMG_SIZE[1],3), 'n_classes':np.shape(test_y)[-1]}

trainer = Trainer(KSAC_network, mconf, train_dataset, test_dataset, tconf)

print('beginning training')
trainer.train()