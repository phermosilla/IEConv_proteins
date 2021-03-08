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

#ifndef BASIS_PROJ_GRADS_CUH_
#define BASIS_PROJ_GRADS_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the gradients of a basis projection operation.
     *  @param  pBasisType                  Type of basis functions used.
     *  @param  pNumPts                     Number of points.
     *  @param  pNumSamples                 Number of samples.
     *  @param  pNumNeighbors               Number of neighbors.
     *  @param  pNumInFeatures              Number of input features.
     *  @param  pInPtsGPUPtr                Input gpu pointer to the array
     *      with the points.
     *  @param  pInPtFeaturesGPUPtr         Input gpu pointer to the array 
     *      with the point features.
     *  @param  pInSamplesGPUPtr            Input gpu pointer to the array
     *      with the samples.
     *  @param  pInNeighborsGPUPtr          Input gpu pointer with the list
     *      of neighbors.
     *  @param  pInSampleNeighIGPUPtr       Input gpu pointer with the 
     *      last neighboring point for each sample.
     *  @param  pInInvRadiiGPUPtr           Input gpu pointer with the 
     *      inverse of the radius used in each dimension.
     *  @param  pInBasisGPUPtr              Input gpu pointer with the basis 
     *      functions.
     *  @param  pInPDFsGPUPtr               Input gpu pointer with the
     *      pdf values for each neighbor.
     *  @param  pInXNeighValGPUPtr          Input gpu pointer with the
     *      x neighbor values.
     *  @param  pInGradGPUPtr               Input gpu pointer with the gradients.
     *  @param  pOutFeatGradsGPUPtr         Output gpu pointer in which
     *      the input feature gradients will be stored.
     *  @param  pOutBasisGradsGPUPtr        Output gpu pointer in which 
     *      the gradients of the basis functions will be stored.
     *  @param  pOutPtGradsGPUPtr           Output gpu pointer in which
     *      the gradients of the points will be stored.
     *  @param  pOutSampleGradsGPUPtr       Output gpu pointer in which
     *      the gradients of the samples will be stored.
     *  @param  pOutPDFGradsGPUPtr          Output gpu pointer in which
     *      the gradietns of the pdf will be stored.
     *  @param  pOutXNeighValGradsGPUPtr    Output gpu pointer in which
     *      the gradietns of the x neighbor values will be stored.
     *  @paramt D                       Number of dimensions.
     *  @paramt K                       Number of basis functions.
     *  @paramt U                       Number of values per neighbor.
     */
    template<int D, int K, int U>
    void basis_proj_grads_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const BasisFunctType pBasisType,
        const unsigned int pNumPts,
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
        const float* pInGradGPUPtr,
        float* pOutFeatGradsGPUPtr,
        float* pOutBasisGradsGPUPtr,
        float* pOutPtGradsGPUPtr,
        float* pOutSampleGradsGPUPtr,
        float* pOutPDFGradsGPUPtr,
        float* pOutXNeighValGradsGPUPtr);
}

#endif