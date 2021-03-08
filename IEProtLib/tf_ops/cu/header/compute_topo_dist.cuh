/////////////////////////////////////////////////////////////////////////////
/// \file compute_topo_dist.cuh
///
/// \brief 
///
/// \copyright Copyright (c) 2020 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTE_TOPO_DISTS_CUH_
#define COMPUTE_TOPO_DISTS_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the distance along the topology
     *  of a graph on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pConstEdge              Boolean that indicates if the
     *      edges have a constant value.
     *  @param  pMaxDist                Maximum distance allowed.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumSpatialNeighbors    Number of neighbors.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrNeighbors      Input pointer to the vector of neighbors
     *      on the GPU.
     *  @param  pInGPUPtrTopo           Input pointer to the vector of neighbors
     *      along the graph on the GPU.
     *  @param  pInGPUPtrSampleTopoI    Input pointer to the vector of number of
     *      neighbors on the graph for each sample on the GPU.
     *  @param  pOutGPUPtrDists         Output pointer to the vector of distances  
     *      on the GPU.      
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_topo_dist_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const bool pConstEdge,
        const float pMaxDist,
        const unsigned int pNumSamples,
        const unsigned int pNumSpatialNeighbors,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrTopo,
        const int* pInGPUPtrSampleTopoI,
        float* pOutGPUPtrDists);

}

#endif