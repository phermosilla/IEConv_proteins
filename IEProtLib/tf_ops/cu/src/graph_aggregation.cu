/////////////////////////////////////////////////////////////////////////////
/// \file graph_aggregation.cu
///
/// \brief 
///
/// \copyright Copyright (c) 2020 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "defines.hpp"
#include "math_helper.cuh"
#include "cuda_kernel_utils.cuh"

#include "graph_aggregation.cuh"

///////////////////////// GPU

__global__ void compute_graph_aggregation_gpu_kernel(
    const bool pNormalize,
    const unsigned int pNumNodes,
    const unsigned int pNumFeatures,
    const unsigned int pNumNeighbors,
    const float* __restrict__ pInGPUPtrFeatures,
    const int2* __restrict__ pInGPUPtrNeighbors,
    const int* __restrict__ pInGPUPtrStartIds,
    float* __restrict__ pOutGPUPtrFeatures)
{
    int initIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the nodes.
    for(int curIndex = initIndex; 
        curIndex < pNumNodes*pNumFeatures; 
        curIndex += totalThreads)
    {
        //Get the ids.
        int nodeId = curIndex/pNumFeatures;
        int featureOffset = curIndex%pNumFeatures;

        //Get the neighbor range.
        int startIndex = 0;
        int endIndex = pInGPUPtrStartIds[nodeId];
        if(nodeId > 0)
            startIndex = pInGPUPtrStartIds[nodeId-1];

        //Iterate over the neighbors.
        float curVal = 0.0f;
        for(int neighCurIndex = startIndex;
            neighCurIndex < endIndex;
            ++neighCurIndex)
        {
            int nodeNeighId = pInGPUPtrNeighbors[neighCurIndex].x;
            curVal += pInGPUPtrFeatures[nodeNeighId*pNumFeatures + featureOffset];
        }

        //Add the final value.
        curVal += pInGPUPtrFeatures[nodeId*pNumFeatures + featureOffset];

        //Normalize if necessary.
        if(pNormalize)
            curVal /= float(endIndex-startIndex+1);

        //Store the result.
        pOutGPUPtrFeatures[nodeId*pNumFeatures + featureOffset] = curVal;
    }
}

__global__ void compute_graph_aggregation_grads_gpu_kernel(
    const bool pNormalize,
    const unsigned int pNumNodes,
    const unsigned int pNumFeatures,
    const unsigned int pNumNeighbors,
    const float* __restrict__ pInGPUPtrGrads,
    const int2* __restrict__ pInGPUPtrNeighbors,
    const int* __restrict__ pInGPUPtrStartIds,
    float* __restrict__ pOutGPUPtrGrads)
{
    int initIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the nodes.
    for(int curIndex = initIndex; 
        curIndex < pNumNodes*pNumFeatures; 
        curIndex += totalThreads)
    {
        //Get the ids.
        int nodeId = curIndex/pNumFeatures;
        int featureOffset = curIndex%pNumFeatures;

        //Get the neighbor range.
        int startIndex = 0;
        int endIndex = pInGPUPtrStartIds[nodeId];
        if(nodeId > 0)
            startIndex = pInGPUPtrStartIds[nodeId-1];

        //Get the input gradient.
        float inGrad = pInGPUPtrGrads[nodeId*pNumFeatures + featureOffset];

        //Normalize if necessary.
        if(pNormalize)
            inGrad /= float(endIndex-startIndex+1);

        //Iterate over the neighbors.
        for(int neighCurIndex = startIndex;
            neighCurIndex < endIndex;
            ++neighCurIndex)
        {
            int nodeNeighId = pInGPUPtrNeighbors[neighCurIndex].x;
            atomicAdd(&pOutGPUPtrGrads[nodeNeighId*pNumFeatures + featureOffset], inGrad);
        }

        //Add the final value.
        atomicAdd(&pOutGPUPtrGrads[nodeId*pNumFeatures + featureOffset], inGrad);
    }
}

///////////////////////// CPU

void mccnn::compute_graph_aggregation_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const bool pNormalize,
    const unsigned int pNumNodes,
    const unsigned int pNumFeatures,
    const unsigned int pNumNeighbors,
    const float* pInGPUPtrFeatures,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrStartIds,
    float* pOutGPUPtrFeatures)
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
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)compute_graph_aggregation_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumNodes*pNumFeatures;
    execBlocks += ((pNumNodes*pNumFeatures)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    compute_graph_aggregation_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNormalize, pNumNodes, pNumFeatures, pNumNeighbors,
        pInGPUPtrFeatures, (const int2*)pInGPUPtrNeighbors,
        pInGPUPtrStartIds, pOutGPUPtrFeatures);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### COMPUTE GRAPH AGGREGATION ###\n");
    fprintf(stderr, "Num nodes: %d\n", pNumNodes);
    fprintf(stderr, "Num features: %d\n", pNumFeatures);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

void mccnn::compute_graph_aggregation_grads_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const bool pNormalize,
    const unsigned int pNumNodes,
    const unsigned int pNumFeatures,
    const unsigned int pNumNeighbors,
    const float* pInGPUPtrGradients,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrStartIds,
    float* pOutGPUPtrGradients)
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

    //Initialize the gradient vector.
    pDevice->memset(pOutGPUPtrGradients, 0, sizeof(float)*pNumFeatures*pNumNodes);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)compute_graph_aggregation_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumNodes*pNumFeatures;
    execBlocks += ((pNumNodes*pNumFeatures)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    compute_graph_aggregation_grads_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNormalize, pNumNodes, pNumFeatures, pNumNeighbors,
        pInGPUPtrGradients, (const int2*)pInGPUPtrNeighbors,
        pInGPUPtrStartIds, pOutGPUPtrGradients);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### COMPUTE GRAPH AGGREGATION GRADS ###\n");
    fprintf(stderr, "Num nodes: %d\n", pNumNodes);
    fprintf(stderr, "Num features: %d\n", pNumFeatures);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}
