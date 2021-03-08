'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PointCloud.py

    \brief Pointcloud object.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf

class PointCloud:
    """Class to represent a point cloud.

    Attributes:
        pts_ (float tensor nxd): List of points.
        batchIds_ (int tensor n): List of batch ids associated with the points.
        batchSize_ (int): Size of the batch.
    """

    def __init__(self, pPts, pBatchIds, pBatchSize):
        """Constructor.

        Args:
            pPts (float tensor nxd): List of points.
            pBatchIds (int tensor n): List of batch ids associated with the points.
            pBatchSize (int): Size of the batch.`
        """
        self.pts_ = pPts
        self.batchIds_ = pBatchIds
        self.batchSize_ = pBatchSize

        #Sort the points based on the batch ids in incremental order.
        _, self.sortedIndicesBatch_ = tf.math.top_k(self.batchIds_, 
            tf.shape(self.batchIds_)[0])
        self.sortedIndicesBatch_ = tf.reverse(self.sortedIndicesBatch_, axis = [0])
        

    def __eq__(self, other):
        """Comparison operator.

        Args:
            other (MCPointCloud): Other point cloud.
        Return:
            True if other is equal to self, False otherwise.
        """
        return self.pts_.name == other.pts_.name and \
            self.batchIds_.name == other.batchIds_.name and \
            self.batchSize_ == other.batchSize_


    def __hash__(self):
        """Method to compute the hash of the object.

        Return:
            Unique hash value.
        """
        return hash((self.pts_.name, self.batchIds_.name, self.batchSize_))