from gan_trainer import GanTrainer, GanTrainerConfig
from nn.PGGAN import Prog_Seg_GAN
from utils.indian_dr_dataset_prep import prep_dataset, load_dataset
import tensorflow as tf
import numpy as np

print('ver',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

start_res = (32,32)
target_res = (256,256)
current_res = (32,32)

train_x,train_y,test_x,test_y = load_dataset('E:/indian dr dataset', target_res)

print('train x shape',train_x.shape)

train_dataset = (train_x,train_y)
test_dataset = (test_x,test_y)


print('this version')
tconf = GanTrainerConfig(max_epochs=30, batch_size=16, learning_rate=1e-4, start_res = start_res, target_res = target_res, current_res = current_res,
                      num_passes= 100, save_dir = 'E:/phd stuff/retinal_lesions_tensorflow/results_and_checkpoints/PGGAN')
print('apparent n classes ',np.shape(test_y)[-1])
mconf = {'start_res':start_res[0], 'target_res':target_res[0],'n_classes':np.shape(test_y)[-1]}

trainer = GanTrainer(Prog_Seg_GAN, mconf, train_dataset, test_dataset, tconf)

print('beginning training')
trainer.train()