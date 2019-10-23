import tensorflow as tf
from model_funcs import get_activation


def context_projection(hs, rs, ts, output_dim, is_tanh=False, is_norm=False, layers=1, scale=0.1,
                       act='linear', initializer=tf.truncated_normal_initializer()):
    print("context_projection", layers, scale, is_tanh, act)
    hts = tf.concat([hs, ts], 1)
    activation = get_activation(act)
    for i in range(layers):
        hts = tf.nn.l2_normalize(hts, 1)
        hts = tf.layers.dense(hts, output_dim, kernel_initializer=initializer, bias_initializer=initializer,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                              activity_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                              activation=activation, name='hts' + str(i), reuse=tf.AUTO_REUSE)
    norm_vec = tf.nn.l2_normalize(hts, 1)
    bias = tf.reduce_sum(rs * norm_vec, 1, keepdims=True) * norm_vec
    edge = rs - bias
    if is_norm:
        edge = tf.nn.l2_normalize(edge, 1)
    if is_tanh:
        edge = tf.tanh(edge)
    return edge


def context_compression(hs, rs, ts, output_dim, is_tanh=False, is_norm=False, layers=1, scale=0.01,
                        act='linear', initializer=tf.truncated_normal_initializer()):
    print("context_compression", layers, scale, act)
    left = tf.concat([hs, rs], 1)
    # left = tf.nn.l2_normalize(left, 1)
    right = tf.concat([rs, ts], 1)
    # right = tf.nn.l2_normalize(right, 1)
    activation = get_activation(act)
    for i in range(layers):
        left = tf.nn.l2_normalize(left, 1)
        left = tf.layers.dense(left, output_dim, kernel_initializer=initializer, bias_initializer=initializer,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                               activity_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                               bias_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                               activation=activation, name='left' + str(i), reuse=tf.AUTO_REUSE)

        right = tf.nn.l2_normalize(right, 1)
        right = tf.layers.dense(right, output_dim, kernel_initializer=initializer, bias_initializer=initializer,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                                activity_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                                activation=activation, name='right' + str(i), reuse=tf.AUTO_REUSE)
    left = tf.nn.l2_normalize(left, 1)
    right = tf.nn.l2_normalize(right, 1)
    crs = tf.concat([left, right], 1)
    for i in range(layers):
        crs = tf.nn.l2_normalize(crs, 1)
        crs = tf.layers.dense(crs, output_dim, kernel_initializer=initializer, bias_initializer=initializer,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                              activity_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=scale),
                              activation=act, name='crs' + str(i), reuse=tf.AUTO_REUSE)
    if is_norm:
        crs = tf.nn.l2_normalize(crs, 1)
    if is_tanh:
        crs = tf.tanh(crs)
    return crs
