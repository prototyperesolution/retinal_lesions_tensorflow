import tensorflow as tf
def focal_loss(labels, logits, gamma: float = 2.0):
    #q = tf.one_hot(labels, logits.shape[-1])
    q = labels
    lnp = tf.nn.log_softmax(logits)
    p = tf.math.exp(lnp)
    return -tf.reduce_sum(
        q * lnp * tf.clip_by_value((1 - p), 1e-8, 1) ** gamma, -1)