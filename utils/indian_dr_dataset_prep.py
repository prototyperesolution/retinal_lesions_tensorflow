import numpy as np
import cv2
import random
import glob

"""first preparing the segmentation part of the dataset"""
def prep_dataset(path_to_dataset):
    x_train, x_test = [],[]
    y_train, y_test = [],[]
    for file in glob.glob(path_to_dataset+'/A. Segmentation/1. Original Images/a. Training Set/*.jpg'):
        x_train.append(file)
        subject = file.split()[-1][4:-4]
        curr_train = []
        for file_y in glob.glob(path_to_dataset+'/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/*/'+subject+'*.tif'):
            curr_train.append(file_y)
        y_train.append(curr_train)
    for file in glob.glob(path_to_dataset + '/A. Segmentation/1. Original Images/b. Testing Set/*.jpg'):
        x_test.append(file)
        subject = file.split()[-1][4:-4]
        curr_test = []
        for file_y in glob.glob(path_to_dataset + '/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/*/' + subject + '*.tif'):
            curr_test.append(file_y)
        y_test.append(curr_test)

    return(x_train,y_train,x_test,y_test)


def prep_batch(x,y,batch_size, img_size=(128,128)):
    x_imgs, y_imgs = [],[]
    indices = np.array([random.randint(0,len(x)-1) for _ in range(batch_size)])
    for index in indices:
        x_imgs.append(cv2.resize(cv2.cvtColor(cv2.imread(x[index]), cv2.COLOR_BGR2RGB),img_size)/255)
        masks = np.zeros((img_size[0],img_size[1],5))
        i = 0
        for mask in y[index]:
            masks[:,:,i] = cv2.resize(cv2.imread(mask),img_size)[:,:,0]/255
            i += 1
        y_imgs.append(masks)

    return(x_imgs, y_imgs)



x_train, y_train, x_test, y_test = prep_dataset('D:/indian dr dataset')

batch = prep_batch(x_train,y_train,8)