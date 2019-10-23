import random
import time
import tensorflow as tf
import numpy as np

from sklearn import preprocessing


def limit_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, neg_param):
    print("limit_loss")
    with tf.name_scope('limit_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('limit_loss'):
        pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
        neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.maximum(pos_score - tf.constant(pos_margin), 0))
        neg_loss = tf.reduce_sum(tf.maximum(tf.constant(neg_margin) - neg_score, 0))
        loss = pos_loss + neg_param * neg_loss
    return loss


def xavier_init(mat_x, mat_y, name, is_l2=False):
    print("xavier_init")
    embeddings = tf.get_variable(name, shape=[mat_x, mat_y], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    if is_l2:
        return tf.nn.l2_normalize(embeddings, 1) if is_l2 else embeddings
    return embeddings


def embed_init(mat_x, mat_y, name, is_l2=False):
    print("embed_init")
    embeddings = tf.Variable(tf.truncated_normal([mat_x, mat_y]), name=name)
    if is_l2:
        return tf.nn.l2_normalize(embeddings, 1) if is_l2 else embeddings
    return embeddings


def random_unit_embeddings(dim1, dim2, name, is_l2=False):
    print("random_unit_embeddings")
    vectors = list()
    for i in range(dim1):
        vectors.append([random.gauss(0, 1) for j in range(dim2)])
    if is_l2:
        vectors = preprocessing.normalize(np.matrix(vectors))
    return tf.Variable(vectors, dtype=tf.float32, name=name)


def mul(tensor1, tensor2, session, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = tf.matmul(tensor1, tensor2, transpose_b=True)
        if sigmoid:
            res = tf.sigmoid(sim_mat).eval(session=session)
        else:
            res = sim_mat.eval(session=session)
    else:
        res = np.matmul(tensor1.eval(session=session), tensor2.eval(session=session).T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    elif act == "sigmoid":
        return tf.sigmoid
    else:
        raise ValueError("Unsupported activation: %s" % act)



