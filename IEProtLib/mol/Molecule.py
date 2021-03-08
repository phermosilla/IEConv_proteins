'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Molecule.py

    \brief Object to store a molecule.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import copy
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from IEProtLibModule import compute_topo_dist

from IEProtLib.pc import PointCloud
from IEProtLib.graph import Graph

class Molecule:
    """Class to represent a molecule.

    Attributes:
        atomPos_ (float tensor nxd): List of atom positions.
        neighbors_ (int tensor mx2): List of neighbor pairs.
        neighStartIndices_ (int tensor n): List of starting indices for each atom
                in the neighboring list.
        batchIds_ (int tensor n): List of batch ids.
        batchSize_ (int): Size of the batch.
    """

    def __init__(self, pAtomPos, pNeighbors, pNeighStartIndices, 
        pBatchIds, pBatchSize, pNeighbors2 = None, 
        pNeigh2StartIndices = None,
        pTodoConstEdgeVal = True):
        """Constructor.

        Args:
            pAtomPos (float tensor nxd): List of atom positions.
            pNeighbors (int tensor mx2): List of neighbor pairs.
            pNeighStartIndices (int tensor n): List of starting indices for each atom
                in the neighboring list.
            pBatchIds (int tensor n): List of batch ids.
            pBatchSize (int): Size of the batch.
            pTodoConstEdgeVal (bool): Boolean that indicates if the edges
                in the topology have a constant value (1.0) for the computations
                of geodesic distances.
        """
        self.atomPos_ = pAtomPos
        self.neighbors_ = pNeighbors
        self.neighStartIndices_ = pNeighStartIndices
        self.batchIds_ = pBatchIds
        self.batchSize_ = pBatchSize
        self.topoConstEdgeVal_ = pTodoConstEdgeVal

        self.pc_ = PointCloud(pAtomPos, pBatchIds, pBatchSize)
        self.graph_ = Graph(pNeighbors, pNeighStartIndices)
        if pNeighbors2 is None:
            self.graph2_ = None
        else:
            self.graph2_ = Graph(pNeighbors2, pNeigh2StartIndices)

    
    def compute_topo_distance(self, pNeighborhood, pThreshold):
        """Method to compute the topological distance along the graph.

        Args:
            pNeighborhood (Neighborhood): Neighborhood.
            pThreshold (float): Threshold.

        Returns:
            (float tensor nxt): Returns the topological distance for each graph.
        """
        if self.topoConstEdgeVal_:
            curThreshold = 6.0
        else:
            curThreshold = pThreshold

        topoDists = compute_topo_dist(self.graph_, pNeighborhood, curThreshold, self.topoConstEdgeVal_)
        topoDists = topoDists / (curThreshold)

        topoDists = tf.reshape(topoDists, [-1, 1])
        if not(self.graph2_ is None):
            topoDists2 = compute_topo_dist(self.graph2_, pNeighborhood, curThreshold, self.topoConstEdgeVal_)
            topoDists2 = topoDists2 / (curThreshold)
            topoDists2 = tf.reshape(topoDists2, [-1, 1])
            topoDists = tf.concat([topoDists, topoDists2], axis=-1)

        return topoDists


class MoleculePH(Molecule):
    """Class to represent a molecule using place holders."""

    def __init__(self, pNumDims, pBatchSize):
        """Constructor.

        Args:
            pNumDims (int): Number of dimensions.
            pBatchSize (int): Size of the batch.
        """
        self.atomPosPH_ = tf.placeholder(tf.float32, [None, pNumDims])
        self.atomNeighsPH_ = tf.placeholder(tf.int32, [None, 2])
        self.atomNeighsStartIdsPH_ = tf.placeholder(tf.int32, [None])
        self.atomTypesPH_ = tf.placeholder(tf.int32, [None])
        self.atomBatchIdsPH_ = tf.placeholder(tf.int32, [None])
        
        Molecule.__init__(self, 
            self.atomPosPH_,
            self.atomNeighsPH_,
            self.atomNeighsStartIdsPH_,
            self.atomBatchIdsPH_,
            pBatchSize)


    def update_dictionary(self, pCurDic, pMoleculeBatch):
        """Method to associate a list of molecules with the placeholders.

        Args:
            pCurDic (dictionary): Output dictionary.
            pMoleculeBatch (tuple with the molecules information): Batch of molecules.
        """
        pCurDic[self.atomPosPH_] = pMoleculeBatch.atomPos_
        pCurDic[self.atomNeighsPH_] = pMoleculeBatch.atomNeighs_
        pCurDic[self.atomNeighsStartIdsPH_] = pMoleculeBatch.atomNeighsStartIds_
        pCurDic[self.atomTypesPH_] = pMoleculeBatch.atomTypes_
        pCurDic[self.atomBatchIdsPH_] = pMoleculeBatch.atomBatchIds_
