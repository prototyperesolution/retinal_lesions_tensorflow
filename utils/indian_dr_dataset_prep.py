import numpy as np
import cv2
import random
import glob
"""this version calls each item from memory. it is very slow as it is constantly loading things in. Might be worth using in situations where
RAM is a consideration"""
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


def prep_batch(x,y,batch_size, img_size=(256,256)):
    x_imgs, y_imgs = [],[]
    indices = np.array([random.randint(0,len(x)-1) for _ in range(batch_size)])
    """some samples have different segmentation results available, so need to do something consistent with them"""
    classes = ['MA','HE','EX','SE','OD']
    for index in indices:
        x_imgs.append(cv2.resize(cv2.cvtColor(cv2.imread(x[index]), cv2.COLOR_BGR2RGB),img_size)/255)
        masks = np.zeros((img_size[0],img_size[1],5))
        for mask in y[index]:
            ID = mask[-6:-4]
            masks[:,:,classes.index(ID)] = cv2.resize(cv2.cvtColor(cv2.imread(mask),cv2.COLOR_BGR2RGB),img_size)[:,:,0]/255
        y_imgs.append(masks)


    return(np.array(x_imgs), np.array(y_imgs))

"""following loads all masks and images into numpy arrays which can then be accessed faster"""

def load_dataset(path_to_dataset, img_size):
    x_train_fn, x_test_fn = [], []
    y_train_fn, y_test_fn = [], []
    x_train_img, y_train_img = [],[]
    x_test_img, y_test_img = [],[]
    classes = ['MA','HE','EX','SE','OD']


    for file in glob.glob(path_to_dataset + '/A. Segmentation/1. Original Images/a. Training Set/*.jpg'):
        x_train_fn.append(file)
        subject = file.split()[-1][4:-4]
        curr_train_fn = []
        for file_y in glob.glob(
                path_to_dataset + '/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/*/' + subject + '*.tif'):
            curr_train_fn.append(file_y)
        y_train_fn.append(curr_train_fn)
    for file in glob.glob(path_to_dataset + '/A. Segmentation/1. Original Images/b. Testing Set/*.jpg'):
        x_test_fn.append(file)
        subject = file.split()[-1][4:-4]
        curr_test_fn = []
        for file_y in glob.glob(
                path_to_dataset + '/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/*/' + subject + '*.tif'):
            curr_test_fn.append(file_y)
        y_test_fn.append(curr_test_fn)

    """creating train dataset"""

    indices = [i for i in range(len(x_train_fn))]
    for index in indices:
        x_train_img.append(cv2.resize(cv2.cvtColor(cv2.imread(x_train_fn[index]), cv2.COLOR_BGR2RGB),img_size)/255)
        masks = np.zeros((img_size[0],img_size[1],5))
        for mask in y_train_fn[index]:
            ID = mask[-6:-4]
            masks[:,:,classes.index(ID)] = cv2.resize(cv2.cvtColor(cv2.imread(mask),cv2.COLOR_BGR2RGB),img_size)[:,:,0]/255
        y_train_img.append(masks)

    """creating test dataset"""
    indices = [i for i in range(len(x_test_fn))]
    for index in indices:
        x_test_img.append(cv2.resize(cv2.cvtColor(cv2.imread(x_test_fn[index]), cv2.COLOR_BGR2RGB), img_size) / 255)
        masks = np.zeros((img_size[0], img_size[1], 5))
        for mask in y_test_fn[index]:
            ID = mask[-6:-4]
            masks[:, :, classes.index(ID)] = cv2.resize(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB), img_size)[:,
                                             :, 0] / 255
        y_test_img.append(masks)

    return np.array(x_train_img), np.array(y_train_img), np.array(x_test_img), np.array(y_test_img)


def augment_data(image, mask, img_size):
    """augments the data by cropping and flipping randomly"""
    flip = np.random.choice([True, False])
    crop = np.random.choice([True, False])
    image_dim = len(image)

    if crop:
        x1, x2 = np.random.randint(0, image_dim), np.random.randint(0, image_dim)
        left = np.min([x1, x2])
        right = np.max([x1, x2])
        if right - left >= img_size[0]:
            image = image[left:right, left:right]
            mask = mask[left:right, left:right]

    if flip:
        axis_to_flip = random.choice([0, 1])
        image = np.flip(image, axis_to_flip)
        mask = np.flip(mask, axis_to_flip)

    image = cv2.resize(image, img_size)
    mask = cv2.resize(mask, img_size)

    return image, mask

def load_batch(x, y, index, batch_size, img_size, augment = False):
    imgs, masks = [],[]
    for i in range(batch_size):
        if augment:
            new_data = augment_data(x[(index+i)%len(x)], y[(index+i)%len(y)], img_size)
            imgs.append(new_data[0])
            masks.append(new_data[1])
        else:
            imgs.append(x[(index+i)%len(x)])
            masks.append(y[(index+i)%len(y)])
    return np.array(imgs), np.array(masks)


