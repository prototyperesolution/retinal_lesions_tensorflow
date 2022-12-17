from nn.train.trainer import Trainer, TrainerConfig
from nn.create_ksac_network import KSAC_network
from utils.indian_dr_dataset_prep import prep_dataset


train_x,train_y,test_x,test_y = prep_dataset('D:/indian dr dataset')

train_dataset = (train_x,train_y)
test_dataset = (test_x,test_y)

print(train_y)

tconf = TrainerConfig(max_epochs=100, batch_size=8, learning_rate=1e-4, batches_per_epoch=200)
mconf = {'input_shape':(128,128,3), 'n_classes':len(test_y[0])}

trainer = Trainer(KSAC_network, mconf, train_dataset, test_dataset, tconf)

#trainer.train()