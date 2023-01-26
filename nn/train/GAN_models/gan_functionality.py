import tensorflow as tf
import numpy as np
from nn.train.GAN_models.gan_layers import build_conv_block, build_depthwise_block, build_transpose_block
from utils import log2, one_hot_encode, resize_image
from scipy.stats import entropy
from nn.train.GAN_models.PGGAN import Prog_Seg_GAN
import matplotlib.pyplot as plt
import cv2

def infer(model, sample, resolution):
    test_img = tf.convert_to_tensor(sample)
    test_img = resize_image(resolution, test_img)
    result_mask = model(test_img).numpy().squeeze()
    #result_one_hot = one_hot_encode(result_mask)
    return result_mask

