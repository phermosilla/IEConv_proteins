/////////////////////////////////////////////////////////////////////////////
/// Copyright 2020 Google LLC
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///    https://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
/////////////////////////////////////////////////////////////////////////////
/// Modifications: pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef BASIS_PROJ_CUH_
#define BASIS_PROJ_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute a monte carlo convolution using an kernel sample.
     *  @param  pBasisType              Type of basis functions used.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pNumInFeatures          Number of input features.
     *  @param  pInPtsGPUPtr            Input gpu pointer to the array
     *      with the points.
     *  @param  pInPtFeaturesGPUPtr     Input gpu pointer to the array
     *      with the input features.
     *  @param  pInSamplesGPUPtr        Input gpu pointer to the array
     *      with the samples.
     *  @param  pInNeighborsGPUPtr      Input gpu pointer with the list
     *      of neighbors.
     *  @param  pInSampleNeighIGPUPtr   Input gpu pointer with the 
     *      last neighbor index for each sample.
     *  @param  pInInvRadiiGPUPtr       Input gpu pointer with the 
     *      inverse of the radius used in each dimension.
     *  @param  pInBasisGPUPtr          Input gpu pointer with the basis
     *      functions.
     *  @param  pInPDFsGPUPtr           Input gpu pointer with the
     *      pdf values for each neighbor.
     *  @param  pInXNeighValGPUPtr      Input gpu pointer with the
     *      x neighbor values.
     *  @param  pOutFeaturesGPUPtr      Output gpu pointer in which
     *      the new features will be stored.
     *  @paramt D                       Number of dimensions.
     *  @paramt K                       Number of basis functions.
     *  @paramt U                       Number of values per neighbor.
     */
    template<int D, int K, int U>
    void basis_proj_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const BasisFunctType pBasisType,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const unsigned int pNumInFeatures,
        const float* pInPtsGPUPtr,
        const float* pInPtFeaturesGPUPtr,
        const float* pInSamplesGPUPtr,
        const int* pInNeighborsGPUPtr,
        const int* pInSampleNeighIGPUPtr,
        const float* pInInvRadiiGPUPtr,
        const float* pInBasisGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInXNeighValGPUPtr,
        float*  pOutFeaturesGPUPtr);
}

#endif