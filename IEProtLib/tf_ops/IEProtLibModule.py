'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file IEProtLibModule.py

    \brief IEProtLib module file.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "build")

sys.path.append(BASE_DIR)

IEProtLib_module = tf.load_op_library(os.path.join(BUILD_DIR, 'IEProtLib.so'))

def compute_keys(pPointCloud, pAABB, pNumCells, pCellSize):
    return IEProtLib_module.compute_keys(
        pPointCloud.pts_,
        pPointCloud.batchIds_, 
        pAABB.aabbMin_/pCellSize,
        pNumCells,
        tf.math.reciprocal(pCellSize))
tf.NoGradient('ComputeKeys')


def build_grid_ds(pKeys, pNumCells, pBatchSize):
    return IEProtLib_module.build_grid_ds(
        pKeys,
        pNumCells,
        pNumCells,
        pBatchSize)
tf.NoGradient('BuildGridDs')


def find_neighbors(pGrid, pPCSamples, pRadii, pMaxNeighbors):    
    return IEProtLib_module.find_neighbors(
        pPCSamples.pts_,
        pPCSamples.batchIds_,
        pGrid.sortedPts_,
        pGrid.sortedKeys_,
        pGrid.fastDS_,
        pGrid.numCells_,
        pGrid.aabb_.aabbMin_/pGrid.cellSizes_,
        tf.math.reciprocal(pGrid.cellSizes_),
        tf.math.reciprocal(pRadii),
        pMaxNeighbors)
tf.NoGradient('FindNeighbors')


def compute_topo_dist(pGraph, pNeighborhood, pMaxDistance, pConstEdge = False):
    intConstEdge = 0
    if pConstEdge:
        intConstEdge = 1
    return IEProtLib_module.compute_topo_dist(
        pNeighborhood.pcSamples_.pts_,
        pNeighborhood.originalNeighIds_,
        pGraph.neighbors_,
        pGraph.nodeStartIndexs_,
        pMaxDistance,
        intConstEdge)
tf.NoGradient('ComputeTopoDist')


def compute_protein_pooling(pGraph):
    return IEProtLib_module.protein_pooling(
        pGraph.neighbors_,
        pGraph.nodeStartIndexs_)
tf.NoGradient('ProteinPooling')


def compute_graph_aggregation(pGraph, pFeatures, pNormalize):
    if pNormalize:
        inNorm = 1
    else:
        inNorm = 0
    return IEProtLib_module.graph_aggregation(
        pFeatures, pGraph.neighbors_,
        pGraph.nodeStartIndexs_, inNorm)
@tf.RegisterGradient("GraphAggregation")
def _compute_graph_aggregation_grad(op, *grads):
    outGrads = IEProtLib_module.graph_aggregation_grads(
        grads[0], op.inputs[1], op.inputs[2],
        op.get_attr("normalize"))
    return [outGrads, None, None]


def basis_proj_bilateral(pNeighborhood, pNeighVals, pInFeatures,
        pBasis, pBasisType, pPtGrads):  
    auxTensor = tf.ones_like(pNeighborhood.neighbors_[:, 0], dtype=tf.float32)
    return IEProtLib_module.basis_proj_bil(
        pNeighborhood.grid_.sortedPts_,
        pInFeatures,
        pNeighborhood.pcSamples_.pts_,
        pNeighborhood.neighbors_,
        pNeighborhood.samplesNeighRanges_, 
        tf.math.reciprocal(pNeighborhood.radii_),
        auxTensor,
        pNeighVals,
        pBasis, 
        pBasisType,
        pPtGrads)
@tf.RegisterGradient("BasisProjBil")
def _basis_proj_bilateral_grad(op, *grads):
    if op.get_attr("pt_grads"):
        featGrads, basisGrads, pointGrads, sampleGrads, _, neighGrads = \
            IEProtLib_module.basis_proj_bil_grads_with_pt_grads(
            op.inputs[0], op.inputs[1], op.inputs[2],
            op.inputs[3], op.inputs[4], op.inputs[5],
            op.inputs[6], op.inputs[7], op.inputs[8], 
            grads[0], op.get_attr("basis_type"))
    else:
        pointGrads = None
        sampleGrads = None
        _ = None
        neighGrads = None
        featGrads, basisGrads = IEProtLib_module.basis_proj_bil_grads(
            op.inputs[0], op.inputs[1], op.inputs[2],
            op.inputs[3], op.inputs[4], op.inputs[5],
            op.inputs[6], op.inputs[7], op.inputs[8], 
            grads[0], op.get_attr("basis_type"))
    return [pointGrads, featGrads, sampleGrads, None, None, 
        None, None, neighGrads, basisGrads]
