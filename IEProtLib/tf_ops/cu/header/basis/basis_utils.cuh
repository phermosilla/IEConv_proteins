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

#ifndef BASIS_UTILS_CUH_
#define BASIS_UTILS_CUH_

#include "defines.hpp"
#include "math_helper.cuh"
#include "basis/basis_interface.cuh"

//Definition of the minimum and maximum kernel points.
#define MIN_BASIS 8
#define MAX_BASIS 32

//Definition of the number of which the number of features should be 
// multiple of.
#define MULTIPLE_IN_FEATURES 8

//Macros to declare and call a template function with a variable
//number of dimensions and variable basis functions.
#define DECLARE_TEMPLATE_DIMS_BASIS(Func)  \
    Func(2, 8,  0)                         \
    Func(2, 16, 0)                         \
    Func(2, 32, 0)                         \
    Func(3, 8,  0)                         \
    Func(3, 16, 0)                         \
    Func(3, 32, 0)                         \
    Func(4, 8,  0)                         \
    Func(4, 16, 0)                         \
    Func(4, 32, 0)                         \
    Func(5, 8,  0)                         \
    Func(5, 16, 0)                         \
    Func(5, 32, 0)                         \
    Func(6, 8,  0)                         \
    Func(6, 16, 0)                         \
    Func(6, 32, 0)                         \
    Func(2, 8,  1)                         \
    Func(2, 16, 1)                         \
    Func(2, 32, 1)                         \
    Func(3, 8,  1)                         \
    Func(3, 16, 1)                         \
    Func(3, 32, 1)                         \
    Func(4, 8,  1)                         \
    Func(4, 16, 1)                         \
    Func(4, 32, 1)                         \
    Func(5, 8,  1)                         \
    Func(5, 16, 1)                         \
    Func(5, 32, 1)                         \
    Func(6, 8,  1)                         \
    Func(6, 16, 1)                         \
    Func(6, 32, 1)                         \
    Func(2, 8,  2)                         \
    Func(2, 16, 2)                         \
    Func(2, 32, 2)                         \
    Func(3, 8,  2)                         \
    Func(3, 16, 2)                         \
    Func(3, 32, 2)                         \
    Func(4, 8,  2)                         \
    Func(4, 16, 2)                         \
    Func(4, 32, 2)                         \
    Func(5, 8,  2)                         \
    Func(5, 16, 2)                         \
    Func(5, 32, 2)                         \
    Func(6, 8,  2)                         \
    Func(6, 16, 2)                         \
    Func(6, 32, 2)                         \
    Func(2, 8,  3)                         \
    Func(2, 16, 3)                         \
    Func(2, 32, 3)                         \
    Func(3, 8,  3)                         \
    Func(3, 16, 3)                         \
    Func(3, 32, 3)                         \
    Func(4, 8,  3)                         \
    Func(4, 16, 3)                         \
    Func(4, 32, 3)                         \
    Func(5, 8,  3)                         \
    Func(5, 16, 3)                         \
    Func(5, 32, 3)                         \
    Func(6, 8,  3)                         \
    Func(6, 16, 3)                         \
    Func(6, 32, 3)                         \


#define NEIGH_VALS_CASE_SWITCH(Dim, K, U, Func, ...)    \
    case U:                                             \
        Func<Dim, K, U>(__VA_ARGS__);                   \
        break;

#define BASIS_CASE_SWITCH(Dim, K, Var, Func, ...)                   \
    case K:                                                         \
        switch(Var){                                                \
            NEIGH_VALS_CASE_SWITCH(Dim, K, 0, Func, __VA_ARGS__)    \
            NEIGH_VALS_CASE_SWITCH(Dim, K, 1, Func, __VA_ARGS__)    \
            NEIGH_VALS_CASE_SWITCH(Dim, K, 2, Func, __VA_ARGS__)    \
            NEIGH_VALS_CASE_SWITCH(Dim, K, 3, Func, __VA_ARGS__)    \
        };                                                          \
        break;

#define DIM_CASE_BASIS_SWITCH_CALL(Dim, Var, Var2, Func, ...)       \
    case Dim:                                                       \
        switch(Var){                                                \
            BASIS_CASE_SWITCH(Dim, 8, Var2, Func, __VA_ARGS__)      \
            BASIS_CASE_SWITCH(Dim, 16, Var2, Func, __VA_ARGS__)     \
            BASIS_CASE_SWITCH(Dim, 32, Var2, Func, __VA_ARGS__)     \
        };                                                          \
        break;

#define DIMENSION_BASIS_SWITCH_CALL(Var1, Var2, Var3, Func, ...)        \
    switch(Var1){                                                       \
        DIM_CASE_BASIS_SWITCH_CALL(2, Var2, Var3, Func, __VA_ARGS__)    \
        DIM_CASE_BASIS_SWITCH_CALL(3, Var2, Var3, Func, __VA_ARGS__)    \
        DIM_CASE_BASIS_SWITCH_CALL(4, Var2, Var3, Func, __VA_ARGS__)    \
        DIM_CASE_BASIS_SWITCH_CALL(5, Var2, Var3, Func, __VA_ARGS__)    \
        DIM_CASE_BASIS_SWITCH_CALL(6, Var2, Var3, Func, __VA_ARGS__)    \
    };

namespace mccnn{

    /**
     *  Types of basis functions available.
     */
    enum class BasisFunctType : int { 
        HPROJ_BIL_RELU=6,
        HPROJ_BIL_LRELU=7,
        HPROJ_BIL_ELU=8,
        HPROJ_BIL_EXP=9
    };

    /**
     *  Method to get the number of parameters of each basis function.
     *  @param  pType       Type of basis function.
     *  @param  pDimensions Number of dimensions.
     *  @param  pXNeighVals Number of values per neighbors.
     *  @return Number of parameters of each basis function.
     */
    __forceinline__ unsigned int get_num_params_x_basis(
        BasisFunctType pType,
        const int pDimensions, 
        const int pXNeighVals)
    {
        unsigned int result = 0;
        switch(pType)
        {
            case BasisFunctType::HPROJ_BIL_RELU:
                result = pDimensions+pXNeighVals+1;
                break;
            case BasisFunctType::HPROJ_BIL_LRELU:
                result = pDimensions+pXNeighVals+1;
                break;
            case BasisFunctType::HPROJ_BIL_ELU:
                result = pDimensions+pXNeighVals+1;
                break;
            case BasisFunctType::HPROJ_BIL_EXP:
                result = pDimensions+pXNeighVals+1;
                break;
        }
        return result;
    }

    /**
     *  Method to create an object of a basis projector.
     *  @param  pBasisType  Type of basis function used.
     *  @return Basis projector object.
     */
    template<int D, int K, int U>
    std::unique_ptr<BasisInterface<D, K, U>> 
    basis_function_factory(BasisFunctType pBasisType);
}

#endif