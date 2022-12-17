from nn.train.trainer import Trainer, TrainerConfig
from nn.create_ksac_network import KSAC_network
from utils.indian_dr_dataset_prep import prep_dataset
import tensorflow as tf

print('ver',tf.__version__)

IMG_SIZE = (256,256)

train_x,train_y,test_x,test_y = prep_dataset('D:/indian dr dataset')

train_dataset = (train_x,train_y)
test_dataset = (test_x,test_y)



tconf = TrainerConfig(max_epochs=100, batch_size=8, learning_rate=1e-4, batches_per_epoch=200)
mconf = {'input_shape':(IMG_SIZE[0],IMG_SIZE[1],3), 'n_classes':len(test_y[0])}

trainer = Trainer(KSAC_network, mconf, train_dataset, test_dataset, tconf)

print('beginning training')
trainer.train()