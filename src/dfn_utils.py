#!/usr/bin/env python3
# coding: utf-8

"""
Utility functions for Deep Flare Net
"""

from __future__ import division, absolute_import, print_function, unicode_literals

from datetime import datetime as dt
import numpy as np
import tensorflow as tf


def make_iso8601_filename(prefix="tmp"):
    """
    e.g. tmp_171106T193925.csv.gz
    """
    tstr = dt.now().strftime("%y%m%dT%H%M%S")
    return "{0:s}_{1:s}.csv.gz".format(prefix, tstr)


def generate_random_onehot_vectors(nb_classes, size):
    """
    generate one-hot vectors
    """
    tmp = np.random.random_integers(0, nb_classes - 1, size)
    one_hot_targets = np.eye(nb_classes)[tmp]
    return one_hot_targets


def leaky_relu(X, leak=0.2):
    """
    Notice: leaky relu's author says it does not work
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def TSS(y_true, y_pred):
    """
    TSS

    import tensorflow as tf
    sess = tf.Session()
    a=tf.contrib.metrics.confusion_matrix([1, 0, 0, 0, 0], [1, 0, 1, 0, 0])
    a.eval(session=sess)
    array([[3, 1],
          [0, 1]], dtype=int32)
    a[0][0].eval(session=sess)
    3 -> tn
    a[0][1].eval(session=sess)
    1 -> fp
    """
    confusion_matrix = tf.confusion_matrix(labels=tf.argmax(y_true, 1),
                                           predictions=tf.argmax(y_pred, 1),
                                           num_classes=2,
                                           dtype=tf.float32)
    tp = confusion_matrix[1][1]
    fn = confusion_matrix[1][0]
    fp = confusion_matrix[0][1]
    tn = confusion_matrix[0][0]
    tmp1 = tf.divide(tp, tf.add(tp, fn))
    tmp2 = tf.divide(fp, tf.add(fp, tn))
    tss = tf.subtract(tmp1, tmp2)
    return tss


def weighted_cross_entropy_with_clip(y_true, y_pred, weights):
    """
    Weighted version of cross_entropy_with_clip
    weights = e.g. [1, 2]
    """
    w = tf.constant(weights, dtype=tf.float32)

    return -1.0 * tf.reduce_mean(w * y_true *
                                 tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))


def cross_entropy_with_clip(y_true, y_pred, weights=None):
    """
    A utility function: cross_entropy_with_clip
    weights: class weighting. e.g. [1, 2]
    """
    if weights is not None:
        w = tf.constant(weights, dtype=tf.float32)
        return -1.0 * tf.reduce_mean(w * y_true *
                                     tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
    else:
        return -1.0 * tf.reduce_mean(y_true *
                                     tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))

# def cross_entropy_with_clip(y_true, y_pred):
#     """
#     A utility function: cross_entropy_with_clip
#     """
#     return -1.0 * tf.reduce_mean(y_true *
#                                  tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))



# Local Variables:
# coding: utf-8
# End:
