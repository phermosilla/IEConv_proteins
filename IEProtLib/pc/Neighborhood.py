'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Neighborhood.py

    \brief Neighborhood of a pointcloud object.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import enum
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from IEProtLib.pc import PointCloud

from IEProtLibModule import find_neighbors


class Neighborhood:
    """Class to represent a neighborhood of points.

    Attributes:
        pcSamples_ (PointCloud): Samples point cloud.
        grid_  (Grid): Regular grid data structure.
        radii_ (float tensor d): Radii used to select the neighbors.
        samplesNeighRanges_ (int tensor n): End of the ranges for each sample.
        neighbors_ (int tensor mx2): Indices of the neighbor point and the sample
            for each neighbor.
        pdf_ (float tensor m): PDF value for each neighbor.
    """

    def __init__(self, pGrid, pRadii, pPCSample = None, pMaxNeighbors = 0):
        """Constructor.

        Args:
            pGrid  (Grid): Regular grid data structure.
            pRadii (float tensor d): Radii used to select the neighbors.
            pPCSample (PointCloud): Samples point cloud. If None, the sorted
                points from the grid will be used.
            pMaxNeighbors (int): Maximum number of neighbors per sample.
        """

        #Save the attributes.
        if pPCSample is None:
            self.equalSamples_ = True
            self.pcSamples_ = PointCloud(pGrid.sortedPts_, \
                pGrid.sortedBatchIds_, pGrid.batchSize_)
        else:
            self.equalSamples_ = False
            self.pcSamples_ = pPCSample
        self.grid_ = pGrid
        self.radii_ = pRadii
        self.pMaxNeighbors_ = pMaxNeighbors

        #Find the neighbors.
        self.samplesNeighRanges_, self.neighbors_ = find_neighbors(
            self.grid_, self.pcSamples_, self.radii_, pMaxNeighbors)

        #Original neighIds.
        auxOriginalNeighsIds = tf.gather(self.grid_.sortedIndices_, self.neighbors_[:,0])
        self.originalNeighIds_ = tf.concat([
            tf.reshape(auxOriginalNeighsIds, [-1,1]), 
            tf.reshape(self.neighbors_[:,1], [-1,1])], axis=-1)

    def apply_neighbor_mask(self, pMask):
        """Method to apply a mask to the neighbors.

        Args:
            pMask (bool tensor n): Tensor with a bool element for each neighbor.
                Those which True will remain in the nieghborhood.

        """

        #Compute the new neighbor list.
        indices = tf.reshape(tf.where(pMask), [-1])
        self.neighbors_ = tf.gather(self.neighbors_, indices)
        self.originalNeighIds_ = tf.gather(
            self.originalNeighIds_, indices)
        newNumNeighs = tf.math.unsorted_segment_sum(
            tf.ones_like(self.neighbors_), 
            self.neighbors_[:,1], 
            tf.shape(self.samplesNeighRanges_)[0])
        self.samplesNeighRanges_ = tf.math.cumsum(newNumNeighs)

        #Update the smooth values.
        if not(self.smoothW_ is None):
            self.smoothW_ = tf.gather(self.smoothW_, indices)

        #Update the pdf values.
        if not(self.pdf_ is None):
            self.pdf_ = tf.gather(self.pdf_, indices)
