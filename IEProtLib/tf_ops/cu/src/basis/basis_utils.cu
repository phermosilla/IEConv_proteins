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

#include "defines.hpp"
#include "cuda_kernel_utils.cuh"
#include "math_helper.cuh"
#include "basis/basis_utils.cuh"
//Include different basis functions.
#include "basis/basis_hproj_bilateral.cuh"

template<int D, int K, int U>
std::unique_ptr<mccnn::BasisInterface<D, K, U>> 
mccnn::basis_function_factory(mccnn::BasisFunctType pBasisType)
{
    if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_RELU){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::RELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_LRELU){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::LRELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_ELU){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::ELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_EXP){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::EXP);
    }
    return std::unique_ptr<mccnn::BasisInterface<D, K, U>>(nullptr);
}

#define BASIS_FUNCTION_FACTORY_DECL(D, K, U)                   \
    template std::unique_ptr<mccnn::BasisInterface<D, K, U>>   \
    mccnn::basis_function_factory<D, K, U>                     \
    (mccnn::BasisFunctType pBasisType);

DECLARE_TEMPLATE_DIMS_BASIS(BASIS_FUNCTION_FACTORY_DECL)