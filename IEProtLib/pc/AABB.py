'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file AABB.py

    \brief Bounding box object.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf

class AABB:
    """Class to represent a bounding box.

    Attributes:
        aabbMin_ (float tensor bxd): List of minimum points of the bounding boxes.
        aabbMax_ (float tensor bxd): List of maximum points of the bounding boxes.
        batchSize_ (int): Size of the batch.
    """

    def __init__(self, pPointCloud):
        """Constructor.

        Args:
            pPointCloud (MCPointCloud): Point cloud from which to compute the bounding box.
        """
        self.batchSize_ = pPointCloud.batchSize_
        self.aabbMin_ = tf.math.unsorted_segment_min(pPointCloud.pts_, pPointCloud.batchIds_, self.batchSize_)-1e-9
        self.aabbMax_ = tf.math.unsorted_segment_max(pPointCloud.pts_, pPointCloud.batchIds_, self.batchSize_)+1e-9
        