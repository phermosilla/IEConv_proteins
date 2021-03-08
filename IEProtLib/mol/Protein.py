'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Protein.py

    \brief Object to represent a protein.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from IEProtLibModule import compute_protein_pooling, compute_graph_aggregation

from IEProtLib.mol import Molecule
from IEProtLib.pc import Grid, AABB
from IEProtLib.py_utils import parse_elem_list

class Protein:
    """Class to represent a protein.
    """

    def __init__(self, 
        pAminoInput,
        pAtomPos, 
        pBatchIds,
        pPoolIds,
        pGraph1Neighbors, 
        pGraph1NeighStartIds, 
        pGraph2Neighbors, 
        pGraph2NeighStartIds, 
        pBatchSize,
        pConfig,
        pAminoPos = None,
        pAtomAminoIds = None, 
        pAtomResidueIds = None,
        pNumResidues = None):
        """Constructor.

        Args:
            pAminoInput (bool): Boolean that indicates if the protein is represented at
                aminoacid level.
            pAtomPos (float tensor nxd): List of atom positions.
            pBatchIds (int tensor n): List of batch ids.
            pPoolIds (list int tensor n): List of pool ids for each level.
            pGraph1Neighbors (list int tensor mx2): List of neighbor pairs. 
            pGraph1NeighStartIds (list int tensor n): List of starting indices for each atom
                in the neighboring list.
            pGraph2Neighbors (list int tensor mx2): List of neighbor pairs. 
            pGraph2NeighStartIds (list int tensor n): List of starting indices for each atom
                in the neighboring list.
            pBatchSize (int): Size of the batch.
            pConfig (dictionary): Dictionary with the config parameters.
            pAminoPos (float tensor n'x3): Aminoacid pos.
            pAtomAminoIds (int tensor n): Identifier of the aminoacid per each atom.
            pAtomResidueIds (int tensor n): Identifier of the residue per each atom.
            pNumResidues (int): Number of residues.
        """

        self.poolType_ = []
        self.poolIds_ = []
        
        self.__bio_pooling__(
                pAminoInput = pAminoInput,
                pAtomPos = pAtomPos, 
                pBatchIds = pBatchIds,
                pPoolIds = pPoolIds,
                pGraph1Neighbors = pGraph1Neighbors, 
                pGraph1NeighStartIds = pGraph1NeighStartIds, 
                pGraph2Neighbors = pGraph2Neighbors, 
                pGraph2NeighStartIds = pGraph2NeighStartIds, 
                pBatchSize = pBatchSize,
                pConfig = pConfig,
                pAminoPos = pAminoPos,
                pAtomAminoIds = pAtomAminoIds, 
                pAtomResidueIds = pAtomResidueIds,
                pNumResidues = pNumResidues)


    def __bio_pooling__(self,
        pAminoInput,
        pAtomPos, 
        pBatchIds,
        pPoolIds,
        pGraph1Neighbors, 
        pGraph1NeighStartIds, 
        pGraph2Neighbors, 
        pGraph2NeighStartIds, 
        pBatchSize,
        pConfig,
        pAminoPos = None,
        pAtomAminoIds = None, 
        pAtomResidueIds = None,
        pNumResidues = None):
        """Biological inspired pooling.

        Args:
            pAminoInput (bool): Boolean that indicates if the protein is represented at
                aminoacid level.
            pAtomPos (float tensor nxd): List of atom positions.
            pBatchIds (int tensor n): List of batch ids.
            pPoolIds (list int tensor n): List of pool ids for each level.
            pGraph1Neighbors (list int tensor mx2): List of neighbor pairs. 
            pGraph1NeighStartIds (list int tensor n): List of starting indices for each atom
                in the neighboring list.
            pGraph2Neighbors (list int tensor mx2): List of neighbor pairs. 
            pGraph2NeighStartIds (list int tensor n): List of starting indices for each atom
                in the neighboring list.
            pBatchSize (int): Size of the batch.
            pConfig (dictionary): Dictionary with the config parameters.
            pAminoPos (float tensor n'x3): Aminoacid pos.
            pAtomAminoIds (int tensor n): Identifier of the aminoacid per each atom.
            pAtomResidueIds (int tensor n): Identifier of the residue per each atom.
            pNumResidues (int): Number of residues.
        """
        numBBPooling = int(pConfig['prot.numbbpoolings'])

        self.aminoInput_ = pAminoInput
        self.poolIds_ = [curPoolId for curPoolId in pPoolIds]

        # Save the first molecule object.
        self.molObjects_ = [
            Molecule(
                pAtomPos, 
                pGraph1Neighbors[0], 
                pGraph1NeighStartIds[0], 
                pBatchIds, pBatchSize,
                pGraph2Neighbors[0], 
                pGraph2NeighStartIds[0])]

        # If input not aminoacids.
        if not self.aminoInput_:
            
            # Save side chain poolings.
            curAtomAminoIds = pAtomAminoIds
            curPoolIds = pPoolIds[0]
            newPos = tf.unsorted_segment_mean(
                self.molObjects_[0].atomPos_, curPoolIds,
                tf.shape(pGraph1NeighStartIds[1])[0])
            curAtomAminoIds = tf.unsorted_segment_max(
                curAtomAminoIds, curPoolIds, 
                tf.shape(pGraph1NeighStartIds[1])[0])
            newBatchIds = tf.unsorted_segment_max(
                self.molObjects_[0].batchIds_, curPoolIds,
                tf.shape(pGraph1NeighStartIds[1])[0])
            self.molObjects_.append(
                Molecule(
                newPos, 
                pGraph1Neighbors[1], 
                pGraph1NeighStartIds[1], 
                newBatchIds, pBatchSize,
                pGraph2Neighbors[1], 
                pGraph2NeighStartIds[1]))
            self.poolType_.append('AVG')
                
            # Save aminoacid level.
            aminoBatchIds = tf.unsorted_segment_max(
                    pBatchIds, pAtomAminoIds,
                    tf.shape(pAminoPos)[0])
            self.molObjects_.append(
                Molecule(
                    pAminoPos, 
                    pGraph1Neighbors[-1], 
                    pGraph1NeighStartIds[-1], 
                    aminoBatchIds, pBatchSize,
                    pGraph2Neighbors[-1], 
                    pGraph2NeighStartIds[-1]))
            self.poolIds_.append(curAtomAminoIds)
            self.poolType_.append('AVG')

        # Compute the backbone poolings.
        for curPool in range(numBBPooling):
            selIndices, pooledNeighs, pooledStartIds = \
                compute_protein_pooling(self.molObjects_[-1].graph_)

            newPositions = compute_graph_aggregation(
                self.molObjects_[-1].graph_, 
                self.molObjects_[-1].atomPos_,
                True)
            newPositions = tf.gather(newPositions, selIndices)

            self.poolIds_.append(selIndices)
            self.poolType_.append('GRA')

            selMask = tf.scatter_nd(
                tf.reshape(selIndices, [-1, 1]),
                tf.ones_like(selIndices),
                tf.shape(self.molObjects_[-1].batchIds_))
            pooledIndices = tf.cumsum(selMask)-1

            # Create new graph2.
            newGraph2 = self.molObjects_[-1].graph2_.pool_graph_collapse_edges(
                pooledIndices, tf.shape(selIndices)[0])
            pooledNeighs2 = newGraph2.neighbors_
            pooledStartIds2 = newGraph2.nodeStartIndexs_                

            newBatchIds = tf.gather(self.molObjects_[-1].batchIds_, selIndices)
            self.molObjects_.append(Molecule(
                newPositions, pooledNeighs, pooledStartIds, 
                newBatchIds, pBatchSize,
                pooledNeighs2, pooledStartIds2))

        self.atomAminoIds_ = pAtomAminoIds
        self.atomResidueIds_ = pAtomResidueIds
        self.numResidues_ = pNumResidues


class ProteinPH(Protein):
    """Class to represent a protein using place holders.
    """

    def __init__(self, pNumDims, pBatchSize, 
        pAminoInput, pConfig):
        """Constructor.

        Args:
            pNumDims (int): Number of dimensions.
            pBatchSize (int): Size of the batch.
            pAminoInput (bool): Boolean that indicates if the input is
                at aminoacid level.
            pConfig (dictionary): Dictionary with the configuration parameters.
        """
        self.aminoInput_ = pAminoInput

        self.atomPosPH_ = tf.placeholder(tf.float32, [None, pNumDims])
        self.atomTypesPH_ = tf.placeholder(tf.int32, [None])
        self.atomBatchIdsPH_ = tf.placeholder(tf.int32, [None])

        self.atomPoolIdsPH_ = []
        self.atomGraph1NeighsPH_ = []
        self.atomGraph1NeighsStartIdsPH_ = []
        self.atomGraph2NeighsPH_ = []
        self.atomGraph2NeighsStartIdsPH_ = []

        if not pAminoInput:
            self.aminoPosPH_ = tf.placeholder(tf.float32, [None, pNumDims])
            self.atomAminoIdsPH_ = tf.placeholder(tf.int32, [None])
            self.atomResidueIdsPH_ = tf.placeholder(tf.int32, [None])
            self.numResiduesPH_ = tf.placeholder(tf.int32, ())

            self.atomPoolIdsPH_.append(tf.placeholder(tf.int32, [None]))

            self.atomGraph1NeighsPH_.append(tf.placeholder(tf.int32, [None, 2]))
            self.atomGraph1NeighsStartIdsPH_.append(tf.placeholder(tf.int32, [None]))
            self.atomGraph2NeighsPH_.append(tf.placeholder(tf.int32, [None, 2]))
            self.atomGraph2NeighsStartIdsPH_.append(tf.placeholder(tf.int32, [None]))
            
            self.atomGraph1NeighsPH_.append(tf.placeholder(tf.int32, [None, 2]))
            self.atomGraph1NeighsStartIdsPH_.append(tf.placeholder(tf.int32, [None]))
            self.atomGraph2NeighsPH_.append(tf.placeholder(tf.int32, [None, 2]))
            self.atomGraph2NeighsStartIdsPH_.append(tf.placeholder(tf.int32, [None]))

            self.atomGraph1NeighsPH_.append(tf.placeholder(tf.int32, [None, 2]))
            self.atomGraph1NeighsStartIdsPH_.append(tf.placeholder(tf.int32, [None]))
            self.atomGraph2NeighsPH_.append(tf.placeholder(tf.int32, [None, 2]))
            self.atomGraph2NeighsStartIdsPH_.append(tf.placeholder(tf.int32, [None]))

        else:
            self.aminoPosPH_ = None
            self.atomAminoIdsPH_ = None
            self.atomResidueIdsPH_ = None
            self.numResiduesPH_ = None  

            self.atomGraph1NeighsPH_ = [tf.placeholder(tf.int32, [None, 2])]
            self.atomGraph1NeighsStartIdsPH_ = [tf.placeholder(tf.int32, [None])]
            self.atomGraph2NeighsPH_ = [tf.placeholder(tf.int32, [None, 2])]
            self.atomGraph2NeighsStartIdsPH_ = [tf.placeholder(tf.int32, [None])]      

        resGraph1 = self.atomGraph1NeighsPH_
        resGraph1StartIds = self.atomGraph1NeighsStartIdsPH_
        resGraph2 = self.atomGraph2NeighsPH_
        resGraph2StartIds = self.atomGraph2NeighsStartIdsPH_

        Protein.__init__(self, 
            pAminoInput = pAminoInput,
            pAtomPos = self.atomPosPH_, 
            pBatchIds = self.atomBatchIdsPH_,
            pPoolIds = self.atomPoolIdsPH_,
            pGraph1Neighbors = resGraph1, 
            pGraph1NeighStartIds = resGraph1StartIds, 
            pGraph2Neighbors = resGraph2, 
            pGraph2NeighStartIds = resGraph2StartIds, 
            pBatchSize = pBatchSize,
            pConfig = pConfig,
            pAminoPos = self.aminoPosPH_,
            pAtomAminoIds = self.atomAminoIdsPH_, 
            pAtomResidueIds = self.atomResidueIdsPH_,
            pNumResidues = self.numResiduesPH_)


    def update_dictionary(self, pCurDic, pProteinBatch):
        """Method to associate a list of proteins with the placeholders.

        Args:
            pCurDic (dictionary): Output dictionary.
            pProteinBatch (tuple with the protein information): Batch of proteins.
        """

        pCurDic[self.atomPosPH_] = pProteinBatch.atomPos_
        pCurDic[self.atomTypesPH_] = pProteinBatch.atomTypes_
        pCurDic[self.atomBatchIdsPH_] = pProteinBatch.atomBatchIds_

        if not self.aminoInput_:
            pCurDic[self.aminoPosPH_] = pProteinBatch.aminoPos_
            pCurDic[self.atomAminoIdsPH_] = pProteinBatch.atomAminoIds_
            pCurDic[self.atomResidueIdsPH_] = pProteinBatch.atomResidueIds_
            pCurDic[self.numResiduesPH_] = pProteinBatch.numResidues_
            pCurDic[self.atomPoolIdsPH_[0]] = pProteinBatch.poolingIds_[0]
            
        numLevels = len(self.atomGraph1NeighsPH_)
        for curLevel in range(numLevels):                
            pCurDic[self.atomGraph1NeighsPH_[curLevel]] = pProteinBatch.graph1Neighs_[curLevel]
            pCurDic[self.atomGraph1NeighsStartIdsPH_[curLevel]] = pProteinBatch.graph1NeighsStartIds_[curLevel]
            pCurDic[self.atomGraph2NeighsPH_[curLevel]] = pProteinBatch.graph2Neighs_[curLevel]
            pCurDic[self.atomGraph2NeighsStartIdsPH_[curLevel]] = pProteinBatch.graph2NeighsStartIds_[curLevel]
        