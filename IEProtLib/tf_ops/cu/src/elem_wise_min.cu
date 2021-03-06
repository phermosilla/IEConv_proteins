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
#include "math_helper.cuh"
#include "cuda_kernel_utils.cuh"

#include "elem_wise_min.cuh"

///////////////////////// GPU

/**
 *  GPU kernel to perform element wise minimum operation with a given
 *  maximum     value.
 *  @param      pNumElements            Number of elements in the array.
 *  @param      pMinValue               Minimum value.
 *  @param      pValues                 Input/Output pointer to the vector of 
 *      values in GPU memory
 *  @rparamt    T                       Type of the elements.
 */
 template<class T>
 __global__ void elem_wise_min_value_gpu(
    const unsigned int pNumElements,
    const T pMinValue,
    T* __restrict__ pValues)
{
    //Get the global thread index.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curIndex = initPtIndex; 
        curIndex < pNumElements; 
        curIndex += totalThreads)
    {
        //Perform the max operation.
        pValues[curIndex] = MCCNN_MIN(pValues[curIndex], pMinValue);
    }
}

///////////////////////// CPU

template<class T>
void mccnn::elem_wise_min_value(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumElements,
    const T pMinValue,
    T* pValuesGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

#ifdef DEBUG_INFO
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStream);
#endif

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*4;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)elem_wise_min_value_gpu<T>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumElements/blockSize;
    execBlocks += (pNumElements%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    elem_wise_min_value_gpu<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumElements, pMinValue, pValuesGPUPtr);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### ELEM WISE MIN ###\n");
    fprintf(stderr, "Num elems: %d\n", pNumElements);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

//TEMPLATE INSTANTIATION
template void mccnn::elem_wise_min_value<float>(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumElements,
    const float pMinValue,
    float* pValuesGPUPtr);
template void mccnn::elem_wise_min_value<int>(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumElements,
    const int pMinValue,
    int* pValuesGPUPtr);