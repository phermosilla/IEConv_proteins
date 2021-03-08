'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MolConvBuild.py

    \brief Builder of convolution operations for protein objects.

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

from IEProtLib.pc import AABB
from IEProtLib.pc import PointCloud
from IEProtLib.pc import Grid
from IEProtLib.pc import Neighborhood
from IEProtLib.graph import GraphConvBuilder
from IEProtLib.graph import Graph
from IEProtLib.tf_utils import BN_AF_DO
from IEProtLib.tf_utils.ConvBuilder import ConvBuilder
from IEProtLib.mol import MolConv
from IEProtLib.mol import Molecule


class MolConvBuilder(ConvBuilder):
    """Class to create convolutions.

    Attributes:
        pcConvBuilder_ (PCConvBuilder): Convolution builder for points.
        graphConvBuilder_ (GraphConvBuilder): Convolution builder for graphs.
    """

    def __init__(self, 
        pWeightRegCollection = "weight_regularization_collection"):
        """Constructor.

        Args:
            pWeightRegCollection (string): Weight regularization collection name.       
        """
        super(MolConvBuilder, self).__init__(pWeightRegCollection)

        self.graphConvBuilder_ = GraphConvBuilder(pWeightRegCollection)
        self.molConvFactory_ = MolConv()


    def create_prot_pooling(self, pInFeatures, pProtein, pLevel, pBNAFDO):
        """Method to create a protein pooling operation.

        Args:
            pInFeatures (float tensor nxf): Input features.
            pProtein (Protein): Protein.
            pLevel (int): Level we want to pool.
            pBNAFDO (BNAFDO): BNAFDO object.
        Returns:
            (float tensor n'xf): Pooled features to the level pLevel+1.
        """

        if pProtein.poolType_[pLevel-1] == "GRA":

            poolFeatures = self.graphConvBuilder_.create_graph_aggregation(
                pInFeatures = pInFeatures,
                pGraph = pProtein.molObjects_[pLevel-1].graph_, 
                pNormalize = True, 
                pSpectralApprox = False)
            poolFeatures = tf.gather(poolFeatures, pProtein.poolIds_[pLevel-1])
        
        elif pProtein.poolType_[pLevel-1] == "AVG":

            poolFeatures = tf.unsorted_segment_mean(pInFeatures,
                pProtein.poolIds_[pLevel-1],
                tf.shape(pProtein.molObjects_[pLevel].batchIds_)[0])

        elif pProtein.poolType_[pLevel-1].startswith("GRAPH_DROP"):

            maskValueBool, poolFeatures, newGraph = self.graphConvBuilder_.create_graph_node_pooling(
                "Graph_drop_pooling_"+str(pLevel), 
                pProtein.molObjects_[pLevel-1].batchIds_,
                pProtein.molObjects_[pLevel-1].graph_, 
                pInFeatures, 
                pProtein.molObjects_[pLevel-1].batchSize_,
                0.5, pBNAFDO)

            newPos = tf.boolean_mask(pProtein.molObjects_[pLevel-1].pc_.pts_, maskValueBool)
            newBatchIds = tf.boolean_mask(pProtein.molObjects_[pLevel-1].batchIds_, maskValueBool)

            if pProtein.molObjects_[pLevel-1].graph2_ is None:
                newGraph2 = Graph(None, None)
            else:
                newGraph2 = pProtein.molObjects_[pLevel-1].graph2_.pool_graph_drop_nodes(
                    maskValueBool, tf.shape(newPos)[0])

            pProtein.molObjects_[pLevel] = Molecule(
                newPos, 
                newGraph.neighbors_, 
                newGraph.nodeStartIndexs_, 
                newBatchIds, 
                pProtein.molObjects_[pLevel-1].batchSize_,
                newGraph2.neighbors_, 
                newGraph2.nodeStartIndexs_)

            if pProtein.poolType_[pLevel-1] == "GRAPH_DROP_AMINO":
                pProtein.poolIds_[pLevel]  = tf.boolean_mask(pProtein.atomAminoIds_, maskValueBool)

        elif pProtein.poolType_[pLevel-1].startswith("GRAPH_EDGE"):
            
            newIndices, poolFeatures, newGraph = self.graphConvBuilder_.create_graph_edge_pooling(
                "Graph_edge_pooling_"+str(pLevel), 
                pProtein.molObjects_[pLevel-1].graph_, 
                pInFeatures, 
                pBNAFDO)

            newPos = tf.unsorted_segment_mean(pProtein.molObjects_[pLevel-1].pc_.pts_,
                newIndices, tf.shape(poolFeatures)[0])
            newBatchIds = tf.unsorted_segment_max(pProtein.molObjects_[pLevel-1].batchIds_,
                newIndices, tf.shape(poolFeatures)[0])

            if pProtein.molObjects_[pLevel-1].graph2_ is None:
                newGraph2 = Graph(None, None)
            else:
                newGraph2 = pProtein.molObjects_[pLevel-1].graph2_.pool_graph_collapse_edges(
                    newIndices, tf.shape(newPos)[0])

            pProtein.molObjects_[pLevel] = Molecule(
                newPos, 
                newGraph.neighbors_, 
                newGraph.nodeStartIndexs_, 
                newBatchIds, 
                pProtein.molObjects_[pLevel-1].batchSize_,
                newGraph2.neighbors_, 
                newGraph2.nodeStartIndexs_)

            if pProtein.poolType_[pLevel-1] == "GRAPH_EDGE_AMINO":
                pProtein.poolIds_[pLevel] = tf.unsorted_segment_max(
                    pProtein.atomAminoIds_, newIndices, tf.shape(poolFeatures)[0])

        return poolFeatures


    def create_molconv_resnet_blocks(self, 
        pMolecule,
        pInFeatures, 
        pNumBlocks, 
        pOutNumFeatures,
        pRadii, 
        pBNAFDO,
        pConvName = None,
        pNumBasis = 32,
        pAtomDropOut = 0.05):
        """Method to create a set of molconv resnet bottleneck blocks.

        Args:
            pMolecule (Molecule): Input molecule.
            pInFeatures (float tensor nxf): Input features.
            pNumBlocks (int): Number of grouping blocks.
            pOutNumFeatures (int): Number of output features.
            pRadii (float): Radius of the convolution.
            pBNAFDO (BRN_AF_DO): Layer to apply batch renorm, activation function,
                and drop out.
            pConvName (string): Name of the convolution. If empty, a unique id is created.
            pNumBasis (int): Number of basis vectors used.
            pAtomDropOut (float): Dropout used during training to randomly discard a
        Return:
            (tensor nxf): Output features.
        """

        #Create the radii tensor.
        radiiTensor = tf.convert_to_tensor(
            np.full((3), pRadii, dtype=np.float32), np.float32)

        #Get the point cloud.
        inPointCloud = pMolecule.pc_

        #Compute the bounding box.
        aabb = AABB(inPointCloud)

        #Compute the grid key.
        grid = Grid(inPointCloud, aabb, radiiTensor)

        #Create the neighborhood.
        neigh = Neighborhood(grid, radiiTensor, inPointCloud, 0)

        #Compute the topological distance.
        topoDists = pMolecule.compute_topo_distance(neigh, pRadii*2.0)

        #Create the convolution name if is not user defined.
        curConvName = pConvName
        if curConvName is None:
            curConvName = hash((neigh, pInFeatures.name))
        
        #Create the different bottleneck blocks.
        curInFeatures = pInFeatures
        for curBlock in range(pNumBlocks):

            #Define a name for the bottleneck block.
            bnName = curConvName+"_resnetb_"+str(curBlock)

            #Save the input features of the block
            blockInFeatures = pBNAFDO(curInFeatures, bnName+"_In_B",
                pApplyBN = True, pApplyNoise = False, pApplyAF = False, pApplyDO = False)

            curInFeatures = pBNAFDO(blockInFeatures, bnName+"_In_AD",
                pApplyBN = False, pApplyNoise = True, pApplyAF = True, pApplyDO = True)

            #Create the first convolution of the block.
            curInFeatures = self.create_1x1_convolution(curInFeatures, pOutNumFeatures//4,
                bnName+"_Conv_1x1_1")
            curInFeatures = pBNAFDO(curInFeatures, bnName+"_Conv_1_BAD",
                pApplyBN = True, pApplyNoise = True, pApplyAF = True, pApplyDO = False)

            #Atom dropout
            tfDORate = tf.cond(pBNAFDO.isTraining_, 
                    true_fn = lambda: pAtomDropOut,
                    false_fn = lambda: 0.0)
            curInFeatures = tf.nn.dropout(curInFeatures, 1.0 - tfDORate, 
                    name=bnName+"_Atom_DO", noise_shape=[tf.shape(curInFeatures)[0], 1])

            #Create the second convolution of the block.
            curInFeatures = tf.gather(curInFeatures, grid.sortedIndices_)
            curInFeatures = self.molConvFactory_.create_convolution( 
                    pConvName = bnName+"_Conv_2",
                    pNeighborhood = neigh, 
                    pTopoDists = topoDists,
                    pFeatures = curInFeatures,
                    pNumOutFeatures = pOutNumFeatures//4,
                    pWeightRegCollection = self.weightRegCollection_,
                    pNumBasis = pNumBasis,
                    pTopoOnly = False,
                    pUse3DDistOnly = True)
            curInFeatures = pBNAFDO(curInFeatures, bnName+"_Conv_2_BAD",
                pApplyBN = True, pApplyNoise = True, pApplyAF = True, pApplyDO = False)

            #Create the third convolution of the block.
            curInFeatures = self.create_1x1_convolution(curInFeatures, pOutNumFeatures,
                bnName+"_Conv_1x1_2")

            #Batch norm.
            curInFeatures = pBNAFDO(curInFeatures, bnName+"_Out_B",
                pApplyBN = True, pApplyNoise = False, pApplyAF = False, pApplyDO = False)

            #If the number of input features is different than the desired output
            if blockInFeatures.get_shape()[-1] != pOutNumFeatures:
                blockInFeatures = pBNAFDO(blockInFeatures, bnName+"_Shortcut_AD",
                    pApplyBN = False, pApplyNoise = True, pApplyAF = True, pApplyDO = True)
                blockInFeatures = self.create_1x1_convolution(blockInFeatures, pOutNumFeatures,
                    bnName+"_Conv_1x1_Shortcut")
                blockInFeatures = pBNAFDO(blockInFeatures, bnName+"_Shortcut_Out_B",
                    pApplyBN = True, pApplyNoise = False, pApplyAF = False, pApplyDO = False)

            #Add the new features to the input features
            curInFeatures = curInFeatures + blockInFeatures

        #Return the resulting features.
        return curInFeatures
