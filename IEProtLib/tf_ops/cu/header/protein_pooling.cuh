/////////////////////////////////////////////////////////////////////////////
/// \file protein_pooling.cuh
///
/// \brief 
///
/// \copyright Copyright (c) 2020 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef PROTEIN_POOLING_CUH_
#define PROTEIN_POOLING_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the starting positions of the backbones in the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumNodes               Number of nodes.
     *  @param  pInGPUPtrNeighbors  Input gpu pointer to the neighbors.
     *  @param  pInGPUPtrStartIds       Input gpu pointer to the starting indices.
     *  @param  pOutGPUPtrStartNodes    Output gpu pointer with 1 in the first 
     *      aminoacid of each backbone.
     */
    void compute_start_backbones_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNodes,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrStartIds,
        int* pOutGPUPtrStartNodes);

    /**
     *  Method to compute the starting positions of the backbones in the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumNodes               Number of nodes.
     *  @param  pInGPUPtrNeighbors  Input gpu pointer to the neighbors.
     *  @param  pInGPUPtrStartIds       Input gpu pointer to the starting indices.
     *  @param  pInGPUPtrBBId           Input/Output gpu pointer to the backbone ids.
     *  @param  pOutGPUPtrBBSIndex      Output gpu pointer with the starting index
     *      of each backbone.
     */
     void store_start_backbones_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNodes,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrStartIds,
        int* pInGPUPtrBBId,
        int* pOutGPUPtrBBSIndex);


    /**
     *  Method to count the number of pooled nodes in the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumNodes               Number of nodes.
     *  @param  pInGPUPtrNeighbors      Input gpu pointer to the neighbors.
     *  @param  pInGPUPtrStartIds       Input gpu pointer to the starting indices.
     *  @param  pInGPUPtrBBSIndexs      Input gpu pointer to the starting indices
     *      of the different backbones.  
     *  @param  pInGPUPtrBBIds          Input gpu pointer to the backbone indices.
     *  @param  pOutGPUPtrIdPooledNode  Output gpu pointer to the array with 1 if
     *      the node is selected and 0 otherwise.
     *  @param  pOutGPUPtrNumNeighs     Output gpu pointer to the array with the 
     *      number of neighbors per pooled point.
     */
     void count_pooled_nodes_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNodes,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrStartIds,
        const int* pInGPUPtrBBSIndexs,
        const int* pInGPUPtrBBIds,
        int* pOutGPUPtrIdPooledNode,
        int* pOutGPUPtrNumNeighs);

    /**
     *  Method to count the number of pooled nodes in the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumNodes               Number of nodes.
     *  @param  pNumPooledNodes         Number of pooled nodes.
     *  @param  pNumPooledNeighs        Number of pooled neighbors.
     *  @param  pInGPUPtrNeighbors      Input gpu pointer to the neighbors.
     *  @param  pInGPUPtrStartIds       Input gpu pointer to the starting indices.
     *  @param  pInGPUPtrPooledNodes    Input gpu pointer to the array with the output
     *      identifier of the pooled nodes.
     *  @param  pInGPUPtrPooledNeighs   Input/Output gpu pointer to the array with the 
     *      output identifiers of the pooled neighbors.
     *  @param  pOutGPUPtrNodeId        Output gpu pointer with the ids of the 
     *      pooled nodes.
     *  @param  pOutGPUPtrNeighs        Output gpu pointer with the neighbors.
     *  @param  pOutGPUPtrNStartIds     Output gpu pointer with the start indices.
     */
     void pool_nodes_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNodes,
        const unsigned int pNumPooledNodes,
        const unsigned int pNumPooledNeighs,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrStartIds,
        const int* pInGPUPtrPooledNodes,
        int* pInGPUPtrPooledNeighs,
        int* pOutGPUPtrNodeId,
        int* pOutGPUPtrNeighs,
        int* pOutGPUPtrNStartIds);
}

#endif
