'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Trainer.py

    \brief Trainer object.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import numpy as np
import configparser

import tensorflow as tf

class Trainer:
    """Class to train a model.

    Attributes:
        optimizerType_ (string): Optimizer type.
        initLR_ (float): Initial learning rate.
        LRDecayFactor_ (float): Learning rate decay factor.
        LRDecayRate_ (int): Learning rate decay rate.
        minLR_ (float): Minimum learning rate allowed.
        wL2Alpha_ (float): Weight used to modulate the weight decay loss.
        weightLoss_ (tensor): Weight decay loss.
        clipGrads_ (float): Value to which the gradients are clipped.
        learningRate_ (tensor): Learning rate. 
        loss_ (tensor): Tensor with the loss value.
        scope_ (string): Scope of the variables.
        optimizer_ (tf optimizer): Tensorflow optimizer.
        trainOps_ (tf operations): Operations to train the model.
    """

    def __init__(self, pConfig, pEpochStep, pLoss, pScope=None, pCheckNans = False):
        """Constructor.

        Args:
            pConfig (dictionary): Dictionary with the configuration of
                the trainer.
            pEpochStep (tensor): Counter with the current epoch id.
            pLoss (tensor): Los we want to minize.
            pScope (string): Scope of the variables we want to train.
            pCheckNans (bool): Boolean that indicates if we check for nan values.
        """

        #Get the configuration of the trainer.
        self.optimizerType_ = pConfig['optimizer']
        self.initLR_ = float(pConfig['initlr'])
        self.LRDecayFactor_ = float(pConfig['lrdecayfactor'])
        self.LRDecayRate_ = int(pConfig['lrdecayrate'])
        self.minLR_ = float(pConfig['minlr'])
        self.wL2Alpha_ = float(pConfig['wl2alpha'])
        self.clipGrads_ = float(pConfig['clipgrads'])
        self.scope_ = pScope

        #Compute the weight regularization loss.
        if self.scope_ is None:
            weights = tf.get_collection("weight_regularization_collection")
        else:
            weights = tf.get_collection("weight_regularization_collection",
                scope=self.scope_)
        if len(weights) > 0:
            self.weightLoss_ = tf.add_n([tf.nn.l2_loss(w) for w in weights])
        else:
            self.weightLoss_ = 0.0

        #Compute the final loss.
        self.loss_ = pLoss + self.weightLoss_*self.wL2Alpha_

        #Create the learning rate.
        self.learningRate_ = tf.train.exponential_decay(self.initLR_, pEpochStep, 
            self.LRDecayRate_, self.LRDecayFactor_, staircase=True)
        self.learningRate_ = tf.maximum(self.learningRate_, self.minLR_)

        #Create the optimizer
        if self.optimizerType_ == "adam":
            beta1 = float(pConfig['beta1'])
            beta2 = float(pConfig['beta2'])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate_,
                beta1 = beta1, beta2 = beta2)
        elif self.optimizerType_ == "momentum":
            momentum = float(pConfig['momentum'])
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learningRate_,
                momentum = momentum)

        #Get the variable list.
        if self.scope_ is None:
            varList = tf.trainable_variables()
        else:
            varList = tf.trainable_variables(scope=self.scope_)

        #Get the update operations.
        if self.scope_ is None:
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        else:
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope_)
        
        #Compute the gradients
        self.gradients_ = optimizer.compute_gradients(self.loss_, var_list = varList)

        #Process gradients.
        countNoneGrads = 0
        for grad, var in self.gradients_:
            if grad is None:
                countNoneGrads += 1
        
        if countNoneGrads > 0:
            print("########### WARNING: Some variables do not affect the output.")

        #Clipping gradients
        if self.clipGrads_ > 0.0:
            clipGrads = [(tf.clip_by_norm(grad, self.clipGrads_), var) for grad, var in self.gradients_ if not(grad is None)]
        else:
            clipGrads = self.gradients_

        #Compute statistics of the gradients.
        gradFlatten = tf.abs(tf.concat([tf.reshape(grad, [-1]) for grad, var in self.gradients_ if not(grad is None)], axis=0))
        numZeros = tf.math.count_nonzero(gradFlatten,dtype=tf.float32)
        self.meanGrad_ = tf.reduce_sum(gradFlatten)/numZeros
        self.maxGrad_ = tf.reduce_max(gradFlatten)
        self.minGrad_ = tf.reduce_min(gradFlatten)

        #Gradient check
        if pCheckNans:
            chekList = []
            for grad, var in self.gradients_:
                if not(grad is None):
                    chekList.append(tf.check_numerics(grad, str(var)))
            updateOps = updateOps+chekList
        
        #Apply gradients
        with tf.control_dependencies(updateOps):
            self.trainOps_ = optimizer.apply_gradients(clipGrads)
