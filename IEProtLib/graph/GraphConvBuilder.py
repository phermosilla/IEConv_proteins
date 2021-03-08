'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GraphConvBuilder.py

    \brief Object to create convolutions for graph data.

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from IEProtLibModule import compute_graph_aggregation

from IEProtLib.graph import Graph
from IEProtLib.tf_utils import BN_AF_DO
from IEProtLib.tf_utils.ConvBuilder import ConvBuilder


class GraphConvBuilder(ConvBuilder):
    """Class to create convolutions.
    """

    def __init__(self, 
        pWeightRegCollection = "weight_regularization_collection"):
        """Constructor.

        Args:
            pWeightRegCollection (string): Weight regularization collection name.       
        """
        
        self.graphDict_ = {}

        super(GraphConvBuilder, self).__init__(pWeightRegCollection)


    def create_graph_aggregation(self, pInFeatures, pGraph, pNormalize = True, 
        pSpectralApprox = False):
        """Method to create a global feature pooling operation.

        Args:
            pInFeatures (float tensor nxf): Input features.
            pGraph (Graph): Input graph.
            pNormalize (bool): Boolean that indicates if the aggregation operation
                normalize the output features based on the number of neighboring nodes.
            pSpectralApprox (bool): First order appoximation of localized spectral
                filters Kipf and Welling.
        Return:
            (tensor nxf2): Output features.
        """

        if pSpectralApprox:
            graphKey = pGraph.__hash__()
            if not(graphKey in self.graphDict_):
                onesVals = tf.ones([tf.shape(pInFeatures)[0], 1])
                DVals = compute_graph_aggregation(pGraph, onesVals, False)
                DVals = tf.stop_gradient(tf.pow(DVals, -0.5))
                self.graphDict_[graphKey] = DVals
            else:
                DVals = self.graphDict_[graphKey]
            transFeat = DVals*pInFeatures
            transFeat = compute_graph_aggregation(pGraph, transFeat, False)
            return DVals*transFeat
        else:
            return compute_graph_aggregation(pGraph, pInFeatures, pNormalize)

        
    def create_graph_node_pooling(self, pConvName, pBatchIds, pGraph, pInFeatures, pBatchSize, 
        pPercent, pBNAFDO):
        """"Method to create a graph pooling operation.
        
        Args:
            pConvName (string): Convolution name.
            pBatchIds (int tensor n): tensor with the batch ids of each node.
            pGraph (Graph): Input graph.
            pInFeatures (float tensor nxf): Input features.
            pBatchSize (int): Size of the batch.
            pPercent (float): Percentage of nodes to select.
            pBNAFDO (MC_BRN_AF_DO): Layer to apply batch renorm, activation function,
                and drop out.

        Returns:
            (int tensor n): Indices of the clusters. Only returned if pKeepEdges equal True.
            (bool tensor n): Boolean mask.
            (float tensor n'xf): New features.
            (MCGraph): Output graph.
        """

        # Compute the weights.
        curFeatures = pBNAFDO(pInFeatures, pConvName+"_In_B",
                pApplyBN = True, pApplyNoise = False, 
                pApplyAF = True, pApplyDO = False)
        unNormVals = self.create_1x1_convolution(curFeatures, 1,
                pConvName+"_Node_scores")
        unNormVals = tf.reshape(unNormVals, [-1])
        normVals = unNormVals / (tf.norm(curFeatures, axis = -1)+1e-6)
        sigmoidVals = tf.reshape(tf.math.sigmoid(unNormVals), [-1, 1])

        # Compute the masks.
        _, sortedIndices = tf.math.top_k(normVals, tf.shape(normVals)[0])
        _, invertedIndices = tf.math.top_k(sortedIndices, tf.shape(normVals)[0])
        invertedIndices = tf.reverse(invertedIndices, [0])
        
        sortedBatchIds = tf.gather(pBatchIds, sortedIndices)
        numNodesXBatch = tf.unsorted_segment_sum(
            tf.ones_like(sortedBatchIds),
            sortedBatchIds, pBatchSize)
        numSelectNodesXBatch = tf.maximum(tf.cast(
            tf.cast(numNodesXBatch, dtype=tf.float32)
            *pPercent, dtype=tf.int32),
            tf.ones_like(numNodesXBatch, dtype= tf.int32))
        xNodeMaxSel = tf.gather(numSelectNodesXBatch, sortedBatchIds)
        accumSum = None
        for curBatch in range(pBatchSize):
            auxMask = tf.equal(sortedBatchIds, curBatch)
            auxMaskOnes = tf.cast(auxMask, tf.int32)
            batchSum = tf.cumsum(auxMaskOnes, exclusive=True) * auxMaskOnes
            if accumSum is None:
                accumSum = batchSum
            else:
                accumSum = accumSum + batchSum

        maskValueBool = tf.less(accumSum, xNodeMaxSel)
        maskValueBool = tf.gather(maskValueBool, invertedIndices)
        maskValues = tf.cast(maskValueBool, dtype=tf.int32)
        
        # Compute the pooled features.
        maskedFeatures = tf.boolean_mask(pInFeatures, maskValueBool)* \
            tf.boolean_mask(sigmoidVals, maskValueBool)

        # Create the new graph.
        newGraph = pGraph.pool_graph_drop_nodes(maskValueBool, tf.reduce_sum(maskValues))

        return maskValueBool, maskedFeatures, newGraph 
