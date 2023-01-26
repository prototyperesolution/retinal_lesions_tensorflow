from trainer import Trainer, TrainerConfig
from nn.models.deeplab_models.create_ksac_network import KSAC_network
from utils.indian_dr_dataset_prep import prep_dataset, load_dataset
import tensorflow as tf
import numpy as np

print('ver',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_SIZE = (512,512)

train_x,train_y,test_x,test_y = load_dataset('D:/indian dr dataset', IMG_SIZE)

train_dataset = (train_x,train_y)
test_dataset = (test_x,test_y)


print('this version')
tconf = TrainerConfig(max_epochs=500, batch_size=16, learning_rate=1e-4, img_size = IMG_SIZE,
                      num_passes= 100, save_dir = 'D:/phd stuff/retinal_lesions_tensorflow/results_and_checkpoints/',
                      ckpt_path = 'D:/phd stuff/retinal_lesions_tensorflow/results_and_checkpoints/2022-12-25_16_512/models/epoch_30_trainIoU_0.615_testIoU_0.485.h5')
print('apparent n classes ',np.shape(test_y)[-1])
mconf = {'input_shape':(IMG_SIZE[0],IMG_SIZE[1],3), 'n_classes':np.shape(test_y)[-1]}

trainer = Trainer(KSAC_network, mconf, train_dataset, test_dataset, tconf)

print('beginning training')
trainer.train()