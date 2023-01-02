from gan_trainer import GanTrainer, GanTrainerConfig
from nn.PGGAN import Prog_Seg_GAN
from utils.indian_dr_dataset_prep import prep_dataset, load_dataset
import tensorflow as tf
import numpy as np

print('ver',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

start_res = (64,64)
target_res = (512,512)

train_x,train_y,test_x,test_y = load_dataset('D:/indian dr dataset', IMG_SIZE)

train_dataset = (train_x,train_y)
test_dataset = (test_x,test_y)


print('this version')
tconf = GanTrainerConfig(max_epochs=20, batch_size=128, learning_rate=1e-4, start_res = start_res, target_res = target_res,
                      num_passes= 100, save_dir = 'D:/phd stuff/retinal_lesions_tensorflow/results_and_checkpoints/PGGAN')
mconf = {'start_res':start_res[0], 'target_res':target_res[0],'n_classes':np.shape(test_y)[-1]}

trainer = GanTrainer(Prog_Seg_GAN, mconf, train_dataset, test_dataset, tconf)

print('beginning training')
trainer.train()