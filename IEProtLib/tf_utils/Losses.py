'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Losses.py

    \brief Implementation of different losses.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import numpy as np
import tensorflow as tf

from IEProtLib.pc import AABB
from IEProtLib.pc import Grid

def cross_entropy(pLabels, pLogits, pBatchIds = None, pBatchSize = None, pWeights =None):
    """Method to compute the cross entropy loss.

    Args:
        pLabels (tensor nx1): Tensor with the labels of each object.
        pLogits (tensor nxc): Tensor with the logits to compute the probabilities.
    Returns:
        (tensor): Cross entropy loss.
    """

    labels = tf.cast(pLabels, tf.int64)

    if not(pWeights is None):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=pLogits[0:tf.shape(labels)[0], :], 
            weights=pWeights, reduction=tf.losses.Reduction.NONE)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=pLogits[0:tf.shape(labels)[0], :], 
            name='XEntropy')
    
    if not(pBatchIds is None):
        cross_entropy = tf.math.unsorted_segment_mean(
            tf.reshape(cross_entropy, [-1, 1]), pBatchIds, pBatchSize)    
    
    return tf.reduce_mean(cross_entropy, name='XEntropy_Mean')


def binary_cross_entropy(pLabels, pLogits, pBatchIds = None, pBatchSize = None, pPosWeight =None):
    """Method to compute the binary cross entropy loss.

    Args:
        pLabels (tensor nx1): Tensor with the labels of each object.
        pLogits (tensor nx1): Tensor with the logits to compute the probabilities.
    Returns:
        (tensor): Cross entropy loss.
    """
    labels = tf.reshape(tf.cast(pLabels, tf.float32), [-1])
    logits = tf.reshape(pLogits[0:tf.shape(pLabels)[0]], [-1])

    if pPosWeight is None:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, 
            logits=logits, name='BinaryXEntropy')
    else:
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=labels, 
            logits=logits, name='BinaryXEntropy', pos_weight = pPosWeight)

    if not(pBatchIds is None):
        cross_entropy = tf.math.unsorted_segment_mean(
            tf.reshape(cross_entropy, [-1, 1]), pBatchIds, pBatchSize)    

    return tf.reduce_mean(cross_entropy, name='BinaryXEntropy_Mean')
