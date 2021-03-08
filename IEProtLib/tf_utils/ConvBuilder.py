'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ConvBuilder.py

    \brief Builder object for convolution.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import math
import enum
import numpy as np
import tensorflow as tf

from IEProtLib.tf_utils import BN_AF_DO

class PoolFeatureMode(enum.Enum):
    maximum = 0
    average = 1
    attention = 2

class ConvBuilder:
    """Class to create convolutions.

    Attributes:
        weightRegCollection_ (string): Weight regularization collection name.
        weightSpectralNorm_ (bool): Boolean that indices if we use spectral
                normalization on the weights of the convolutions.
    """

    def __init__(self, 
        pWeightRegCollection = "weight_regularization_collection"):
        """Constructor.

        Args:
            pWeightRegCollection (string): Weight regularization collection name.
        """
        self.weightRegCollection_ = pWeightRegCollection


    def create_1x1_convolution(self,
        pInFeatures,
        pNumOutFeatures,
        pConvName = None,
        pBiases = True,
        pInitializer = None,
        pBiasInitializer = None):
        """Method to create a 1x1 convolution.

        Args:
            pInFeatures (float tensor nxf): Input features.
            pOutNumFeatures (int): Number of output features.
            pConvName (string): Name of the convolution.
            pBiases (bool): Boolean that indicates if we will use biases.
            pInitializer (tensorflow initializer): Weights initializer.
            pBiasInitializer (tensorflow initializer): biases initializer.
        Return:
            (tensor nxf2): Output features.
        """
        #Compute the name
        curConvName = pConvName
        if curConvName is None:
            curConvName = pInFeatures.name.split("/")[-1]
            curConvName = curConvName.split(":")[0]+"_1x1_Conv"

        #If tensor has more than 2 dimensions reshape it.
        currentShape = pInFeatures.shape.as_list()
        if len(currentShape) > 2:
            curInput = tf.reshape(pInFeatures, [-1, currentShape[-1]])
        else:
            curInput = pInFeatures

        #Create the weights.
        numInFeatures = curInput.shape.as_list()[1]
        curInitializer = pInitializer
        if curInitializer is None:
            stdDev = math.sqrt(2.0/float(numInFeatures)) #RELU
            curInitializer = tf.initializers.truncated_normal(stddev=stdDev)
        weights = tf.get_variable(curConvName+'_weights', shape=[numInFeatures, pNumOutFeatures], 
            initializer=curInitializer, dtype=tf.float32, trainable=True)
        tf.add_to_collection(self.weightRegCollection_, tf.reshape(weights, [-1]))

        #Create the biases.
        if pBiases:
            biasInitializer = pBiasInitializer
            if biasInitializer is None:
                biasInitializer = tf.initializers.zeros()
            biases = tf.get_variable(curConvName+'_biases', shape=[1, pNumOutFeatures], 
                initializer=biasInitializer, dtype=tf.float32, trainable=True)

        #Multiply the input.
        outFeatures = tf.matmul(curInput, weights)
        if pBiases:
            outFeatures = outFeatures + biases

        if len(currentShape) > 2:
            currentShape[-1] = pNumOutFeatures
            currentShape = [curShapeElem if not(curShapeElem is None) else -1 for curShapeElem in currentShape]
            outFeatures = tf.reshape(outFeatures, currentShape)

        return outFeatures


    def create_global_feature_pooling(self,
        pInFeatures,
        pBatchIds, 
        pBatchSize,
        pBNAFDO,
        pMode = PoolFeatureMode.maximum):
        """Method to create a global feature pooling operation.

        Args:
            pInFeatures (float tensor nxf): Input features.
            pBatchIds (int tensor n): Input batch ids.
            pBatchSize (int): Siye of the batch.
            pBNAFDO (BRN_AF_DO): Layer to apply batch renorm, activation function,
                and drop out.
            pMode (PoolFeatureMode): Pooling mode.
        Return:
            (tensor bxf2): Output features.
        """

        outFeature = None
        #Perform maximum pooling.
        if pMode == PoolFeatureMode.maximum:
            outFeature = tf.math.unsorted_segment_max(pInFeatures, 
                pBatchIds, pBatchSize)
        #Perform average pooling.
        elif pMode == PoolFeatureMode.average:
            outFeature = tf.math.unsorted_segment_mean(pInFeatures, 
                pBatchIds, pBatchSize)
        #Perform attention pooling.
        elif pMode == PoolFeatureMode.attention:
            #Compute the weights.
            hid1 = pBNAFDO(pInFeatures, "Global_Attention_BAD")
            hid1 = self.create_1x1_convolution(hid1, 1)
            weight = tf.math.exp(tf.tanh(hid1))
            #Sum the weights for each model.
            sumWeights = tf.math.unsorted_segment_sum(hid1, 
                pBatchIds, pBatchSize)
            #Normalize weights (softmax).
            sumGather = tf.gather(sumWeights, pBatchIds)
            weight = weight*tf.math.reciprocal(sumGather)
            #Compute the attention (final model features).
            weightedFeatures = tf.multiply(pInFeatures, weight)
            outFeature = tf.math.unsorted_segment_sum(weightedFeatures, 
                pBatchIds, pBatchSize)

        return outFeature

