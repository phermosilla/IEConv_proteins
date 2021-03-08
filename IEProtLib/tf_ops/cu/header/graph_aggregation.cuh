/////////////////////////////////////////////////////////////////////////////
/// \file graph_aggregation.cuh
///
/// \brief 
///
/// \copyright Copyright (c) 2020 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTE_GRAPH_AGGREGATION_CUH_
#define COMPUTE_GRAPH_AGGREGATION_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute a graph aggregation operation on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNormalize              Boolean that indicates if we
     *      normalize the result dividing by the number of neighbors.
     *  @param  pNumNodes               Number of nodes.
     *  @param  pNumFeatures            Number of features.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pInGPUPtrFeatures       Input gpu pointer to the features.
     *  @param  pInGPUPtrNeighbors      Input gpu pointer to the neighbors.
     *  @param  pInGPUPtrStartIds       Input gpu pointer to the starting indices.
     *  @param  pOutGPUPtrFeatures      Output gpu pointer to the features.
     */
    void compute_graph_aggregation_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const bool pNormalize,
        const unsigned int pNumNodes,
        const unsigned int pNumFeatures,
        const unsigned int pNumNeighbors,
        const float* pInGPUPtrFeatures,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrStartIds,
        float* pOutGPUPtrFeatures);

    /**
     *  Method to compute the gradients of a graph aggregation operation on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNormalize              Boolean that indicates if we
     *      normalize the result dividing by the number of neighbors.
     *  @param  pNumNodes               Number of nodes.
     *  @param  pNumFeatures            Number of features.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pInGPUPtrGradients      Input gpu pointer to the input gradients.
     *  @param  pInGPUPtrNeighbors      Input gpu pointer to the neighbors.
     *  @param  pInGPUPtrStartIds       Input gpu pointer to the starting indices.
     *  @param  pOutGPUPtrGradients     Output gpu pointer to the output gradients.
     */
     void compute_graph_aggregation_grads_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const bool pNormalize,
        const unsigned int pNumNodes,
        const unsigned int pNumFeatures,
        const unsigned int pNumNeighbors,
        const float* pInGPUPtrGradients,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrStartIds,
        float* pOutGPUPtrGradients);
}

#endif