import tensorflow as tf
from utils import resize_image


def infer(model, sample, resolution):
    test_img = tf.convert_to_tensor(sample)
    test_img = resize_image(resolution, test_img)
    result_mask = model(test_img).numpy().squeeze()
    #result_one_hot = one_hot_encode(result_mask)
    return result_mask

