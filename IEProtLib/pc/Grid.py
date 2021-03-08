'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Grid.py

    \brief Object to store a pointcloud in a regular grid.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from IEProtLibModule import compute_keys
from IEProtLibModule import build_grid_ds

class Grid:
    """Class to represent a point cloud distributed in a regular grid.

    Attributes:
        batchSize_ (int): Size of the batch.
        cellSizes_ (float tensor d): Cell size.
        pointCloud_ (PointCloud): Point cloud.
        aabb_ (AABB): AABB.
        numCells_ (int tensor d): Number of cells of the grids.
        curKeys_ (int tensor n): Keys of each point.
        sortedKeys_ (int tensor n): Keys of each point sorted.
        sortedIndices_ (int tensor n): Original indices to the original
            points.
        fastDS_ (int tensor BxCXxCY): Fast access data structure.
    """

    def __init__(self, pPointCloud, pAABB, pCellSizes):
        """Constructor.

        Args:
            pPointCloud (PointCloud): Point cloud to distribute in the grid.
            pAABB (AABB): Bounding boxes.
            pCellSizes (tensor float n): Size of the grid cells in each dimension.
        """
        #Save the attributes.
        self.batchSize_ = pAABB.batchSize_
        self.cellSizes_ = pCellSizes
        self.pointCloud_ = pPointCloud
        self.aabb_ = pAABB

        #Compute the number of cells in the grid.
        aabbSizes = self.aabb_.aabbMax_ - self.aabb_.aabbMin_
        batchNumCells = tf.cast(tf.math.ceil(aabbSizes/self.cellSizes_), tf.int32)
        self.numCells_ = tf.maximum(tf.reduce_max(batchNumCells, axis=0), 1)

        #Compute the key for each point.
        self.curKeys_ = compute_keys(self.pointCloud_, self.aabb_, self.numCells_, 
            self.cellSizes_)

        #Sort the keys.
        self.sortedKeys_, self.sortedIndices_ = tf.math.top_k(self.curKeys_, 
            tf.shape(self.curKeys_)[0])

        #Compute the invert indexs.
        _, ptIndexs = tf.math.top_k(self.sortedIndices_, 
            tf.shape(self.sortedIndices_)[0])
        self.invertedIndices_ = tf.reverse(ptIndexs, [0])

        #Get the sorted points and batch ids.
        self.sortedPts_ = tf.gather(self.pointCloud_.pts_, self.sortedIndices_)
        self.sortedBatchIds_ = tf.gather(self.pointCloud_.batchIds_, self.sortedIndices_)

        #Build the fast access data structure.
        self.fastDS_ = build_grid_ds(self.sortedKeys_, self.numCells_, self.batchSize_)

