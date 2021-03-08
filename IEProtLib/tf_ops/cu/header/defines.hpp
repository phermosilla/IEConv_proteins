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

#ifndef DEFINES_H_
#define DEFINES_H_

#include <memory>

//Definition of the minimum and maximum number of dimensions.
#define MIN_DIMENSIONS 2
#define MAX_DIMENSIONS 6

//Macros to declare and call a template function with a variable
//number of dimensions.
#define DECLARE_TEMPLATE_DIMS(Func) \
    Func(2)                         \
    Func(3)                         \
    Func(4)                         \
    Func(5)                         \
    Func(6)                                              

#define DIMENSION_CASE_SWITCH(Dim, Func, ...)   \
    case Dim:                                   \
        Func<Dim>(__VA_ARGS__);                 \
        break;

#define DIMENSION_SWITCH_CALL(Var, Func, ...)       \
    switch(Var){                                    \
        DIMENSION_CASE_SWITCH(2, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(3, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(4, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(5, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(6, Func, __VA_ARGS__) \
    }  

#define DIMENSION_CASE_SWITCH_RETURN(Dim, Ret, Func, ...)   \
    case Dim:                                   \
        Ret = Func<Dim>(__VA_ARGS__);                 \
        break;

#define DIMENSION_SWITCH_CALL_RETURN(Var, Ret, Func, ...)       \
    switch(Var){                                    \
        DIMENSION_CASE_SWITCH_RETURN(2, Ret, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH_RETURN(3, Ret, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH_RETURN(4, Ret, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH_RETURN(5, Ret, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH_RETURN(6, Ret, Func, __VA_ARGS__) \
    }  


//Definition of the minimum and maximum MLP kernel size.
#define MIN_KERNEL_MLP_SIZE 4
#define MAX_KERNEL_MLP_SIZE 16

//Macros to declare and call a template function with a variable
//number of dimensions and variable MLP size.
#define DECLARE_TEMPLATE_DIMS_MLP(Func) \
    Func(2, 4)                          \
    Func(2, 8)                          \
    Func(2, 16)                         \
    Func(3, 4)                          \
    Func(3, 8)                          \
    Func(3, 16)                         \
    Func(4, 4)                          \
    Func(4, 8)                          \
    Func(4, 16)                         \
    Func(5, 4)                          \
    Func(5, 8)                          \
    Func(5, 16)                         \
    Func(6, 4)                          \
    Func(6, 8)                          \
    Func(6, 16)                         \

#define MLP_CASE_SWITCH(Dim, MLP, Func, ...)    \
    case MLP:                                   \
        Func<Dim, MLP>(__VA_ARGS__);            \
        break;

#define DIM_CASE_MLP_SWITCH_CALL(Dim, Var, Func, ...)   \
    case Dim:                                           \
        switch(Var){                                    \
            MLP_CASE_SWITCH(Dim, 4, Func, __VA_ARGS__)  \
            MLP_CASE_SWITCH(Dim, 8, Func, __VA_ARGS__)  \
            MLP_CASE_SWITCH(Dim, 16, Func, __VA_ARGS__) \
        };                                              \
        break;

#define DIMENSION_MLP_SWITCH_CALL(Var1, Var2, Func, ...)        \
    switch(Var1){                                               \
        DIM_CASE_MLP_SWITCH_CALL(2, Var2, Func, __VA_ARGS__)    \
        DIM_CASE_MLP_SWITCH_CALL(3, Var2, Func, __VA_ARGS__)    \
        DIM_CASE_MLP_SWITCH_CALL(4, Var2, Func, __VA_ARGS__)    \
        DIM_CASE_MLP_SWITCH_CALL(5, Var2, Func, __VA_ARGS__)    \
        DIM_CASE_MLP_SWITCH_CALL(6, Var2, Func, __VA_ARGS__)    \
    };

//Definition of the min and max operation for cuda code.
#define MCCNN_MAX(a, b) (a < b) ? b : a;
#define MCCNN_MIN(a, b) (a > b) ? b : a;

namespace mccnn{
    //Definition of the int 64 bit.
    typedef long long int64_m;
}

//Definition of make unique for C++11 (Only available in C++14)
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif