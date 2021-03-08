/////////////////////////////////////////////////////////////////////////////
/// \file protein_pooling.cu
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

#include "protein_pooling.cuh"

///////////////////////// GPU

__global__ void compute_start_backbones_gpu_kernel(
    const unsigned int pNumNodes,
    const int2* __restrict__ pInGPUPtrNeighbors,
    const int* __restrict__ pInGPUPtrStartIds,
    int* __restrict__ pOutGPUPtrStartNodes)
{
    int initIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the nodes.
    for(int nodeId = initIndex; 
        nodeId < pNumNodes; 
        nodeId += totalThreads)
    {
        //Get the neighbor range.
        int startIndex = 0;
        int endIndex = pInGPUPtrStartIds[nodeId];
        if(nodeId > 0)
            startIndex = pInGPUPtrStartIds[nodeId-1];

        //Store the result.
        if((endIndex-startIndex) == 1){
            int2 neighId = pInGPUPtrNeighbors[startIndex];
            if(neighId.x > neighId.y)
                pOutGPUPtrStartNodes[nodeId] = 1;
            else
                pOutGPUPtrStartNodes[nodeId] = 0;
        }else if((endIndex-startIndex) == 0){
            pOutGPUPtrStartNodes[nodeId] = 1;
        }else
            pOutGPUPtrStartNodes[nodeId] = 0;
    }
}

__global__ void store_start_backbones_gpu_kernel(
    const unsigned int pNumNodes,
    const int2* __restrict__ pInGPUPtrNeighbors,
    const int* __restrict__ pInGPUPtrStartIds,
    int* __restrict__ pInGPUPtrBBIds,
    int* __restrict__ pOutGPUPtrBBSIndex)
{
    int initIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the nodes.
    for(int nodeId = initIndex; 
        nodeId < pNumNodes; 
        nodeId += totalThreads)
    {
        //Get the neighbor range.
        int startIndex = 0;
        int endIndex = pInGPUPtrStartIds[nodeId];
        if(nodeId > 0)
            startIndex = pInGPUPtrStartIds[nodeId-1];

        //Store the result.
        if((endIndex-startIndex) == 1){
            int2 neighId = pInGPUPtrNeighbors[startIndex];
            if(neighId.x > neighId.y){
                int backboneId = pInGPUPtrBBIds[nodeId];
                pOutGPUPtrBBSIndex[backboneId] = nodeId;
                pInGPUPtrBBIds[nodeId] += 1;
            }
        }else if((endIndex-startIndex) == 0){
            int backboneId = pInGPUPtrBBIds[nodeId];
            pOutGPUPtrBBSIndex[backboneId] = nodeId;
            pInGPUPtrBBIds[nodeId] += 1;
        }
    }
}

__global__ void count_pooled_nodes_gpu_kernel(
    const unsigned int pNumNodes,
    const int2* __restrict__ pInGPUPtrNeighbors,
    const int* __restrict__ pInGPUPtrStartIds,
    const int* __restrict__ pInGPUPtrBBStartIndexs,
    const int* __restrict__ pInGPUPtrBBIds,
    int* __restrict__ pOutGPUPtrIdPooledNode,
    int* __restrict__ pOutGPUPtrNumNeighs)
{
    int initIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the nodes.
    for(int nodeId = initIndex; 
        nodeId < pNumNodes; 
        nodeId += totalThreads)
    {
        //Get the identifier of the first element of the backbone.
        int bbId = pInGPUPtrBBIds[nodeId]-1;
        int refId = pInGPUPtrBBStartIndexs[bbId];

        //Check if this node will be pooled.
        bool pooled = (nodeId-refId)%2 == 0;

        //Store the result.
        pOutGPUPtrIdPooledNode[nodeId] = (pooled)?1:0;

        //Get the neighbor range.
        int startIndex = 0;
        int endIndex = pInGPUPtrStartIds[nodeId];
        if(nodeId > 0)
            startIndex = pInGPUPtrStartIds[nodeId-1];

        //Check the neighbors.
        int neighCounter = 0;
        for(int i = startIndex; i < endIndex; ++i)
        {
            //Get the indices.
            int2 curNeigh = pInGPUPtrNeighbors[i];

            //Get the neighbor list of the neighbor.
            int neighStartIndex = 0;
            int neighEndIndex = pInGPUPtrStartIds[curNeigh.x];
            if(curNeigh.x > 0)
                neighStartIndex = pInGPUPtrStartIds[curNeigh.x-1];
            
            //If it has more than one neighbor...
            neighCounter += ((neighEndIndex-neighStartIndex) > 1)?1:0;
        }
        //Store the result.
        pOutGPUPtrNumNeighs[nodeId] = (pooled)?neighCounter:0;
    }
}

__global__ void store_pooled_nodes_gpu_kernel(
    const unsigned int pNumNodes,
    const unsigned int pNumPooledNodes,
    const unsigned int pNumPooledNeighs,
    const int2* __restrict__ pInGPUPtrNeighbors,
    const int* __restrict__ pInGPUPtrStartIds,
    const int* __restrict__ pInGPUPtrPooledNodes,
    int* __restrict__ pInGPUPtrPooledNeighs,
    int* __restrict__ pOutGPUPtrNodeIds,
    int2* __restrict__ pOutGPUPtrNodeNeighs,
    int* __restrict__ pOutGPUPtrStartIds)
{
    int initIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the nodes.
    for(int nodeId = initIndex; 
        nodeId < pNumNodes; 
        nodeId += totalThreads)
    {
        //Get the index and the next index.
        int nextIndex = 0;
        int curIndex = pInGPUPtrPooledNodes[nodeId];
        if(nodeId < pNumNodes-1)
            nextIndex = pInGPUPtrPooledNodes[nodeId+1];
        else
            nextIndex = pNumPooledNodes;

        //If the node is going to be pooled.
        if(curIndex != nextIndex){

            //Store the node id.
            pOutGPUPtrNodeIds[curIndex] = nodeId;

            //Store the end neighbor index.
            if(nodeId < pNumNodes-1)
                pOutGPUPtrStartIds[curIndex] = pInGPUPtrPooledNeighs[nodeId+1];
            else
                pOutGPUPtrStartIds[curIndex] = pNumPooledNeighs;

            //Get the neighbor range.
            int startIndex = 0;
            int endIndex = pInGPUPtrStartIds[nodeId];
            if(nodeId > 0)
                startIndex = pInGPUPtrStartIds[nodeId-1];

            //Check the neighbors.
            for(int i = startIndex; i < endIndex; ++i)
            {
                //Get the indices.
                int2 curNeigh = pInGPUPtrNeighbors[i];

                //Get the neighbor list of the neighbor.
                int neighStartIndex = 0;
                int neighEndIndex = pInGPUPtrStartIds[curNeigh.x];
                if(curNeigh.x > 0)
                    neighStartIndex = pInGPUPtrStartIds[curNeigh.x-1];
                
                //If it has more than one neighbor...
                if((neighEndIndex-neighStartIndex) > 1){

                    //Get the identifier where to store the neighbor.
                    int curNeighIndex = pInGPUPtrPooledNeighs[nodeId];
                    pInGPUPtrPooledNeighs[nodeId] += 1;
                    int newNeighId = -1;
#pragma unroll(2)
                    for(int j = neighStartIndex; j < neighEndIndex; ++j)
                    {
                        //Get the indices.
                        int2 curNeighNeigh = pInGPUPtrNeighbors[j];
                        //If it is a valid neighbor.
                        if(curNeighNeigh.x != nodeId)
                            newNeighId = curNeighNeigh.x;
                    }
                    
                    //Store the neighbor.
                    int transNewNeigh = pInGPUPtrPooledNodes[newNeighId];
                    pOutGPUPtrNodeNeighs[curNeighIndex] = make_int2(
                        transNewNeigh, curIndex);
                }
            }
        }
    }
}

///////////////////////// CPU

void mccnn::compute_start_backbones_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumNodes,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrStartIds,
    int* pOutGPUPtrStartNodes)
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
        blockSize,(const void*)compute_start_backbones_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumNodes;
    execBlocks += (pNumNodes%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    compute_start_backbones_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumNodes, (const int2*)pInGPUPtrNeighbors, pInGPUPtrStartIds, pOutGPUPtrStartNodes);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### COMPUTE START BACKBONES ###\n");
    fprintf(stderr, "Num nodes: %d\n", pNumNodes);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

void mccnn::store_start_backbones_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumNodes,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrStartIds,
    int* pInGPUPtrBBId,
    int* pOutGPUPtrBBSIndex)
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
        blockSize,(const void*)store_start_backbones_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumNodes;
    execBlocks += (pNumNodes%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    store_start_backbones_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumNodes, (const int2*)pInGPUPtrNeighbors, pInGPUPtrStartIds, 
        pInGPUPtrBBId, pOutGPUPtrBBSIndex);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### STORE START BACKBONES ###\n");
    fprintf(stderr, "Num nodes: %d\n", pNumNodes);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

void mccnn::count_pooled_nodes_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumNodes,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrStartIds,
    const int* pInGPUPtrBBSIndexs,
    const int* pInGPUPtrBBIds,
    int* pOutGPUPtrIdPooledNode,
    int* pOutGPUPtrNumNeighs)
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
        blockSize,(const void*)compute_start_backbones_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumNodes;
    execBlocks += (pNumNodes%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    count_pooled_nodes_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumNodes,
        (const int2*)pInGPUPtrNeighbors,
        pInGPUPtrStartIds, pInGPUPtrBBSIndexs, 
        pInGPUPtrBBIds, pOutGPUPtrIdPooledNode,
        pOutGPUPtrNumNeighs);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### COUNT POOLED NODES ###\n");
    fprintf(stderr, "Num nodes: %d\n", pNumNodes);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

void mccnn::pool_nodes_gpu(
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
    int* pOutGPUPtrNStartIds)
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
        blockSize,(const void*)store_pooled_nodes_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumNodes;
    execBlocks += (pNumNodes%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    store_pooled_nodes_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumNodes, pNumPooledNodes, pNumPooledNeighs,
        (const int2*)pInGPUPtrNeighbors,
        pInGPUPtrStartIds, pInGPUPtrPooledNodes, pInGPUPtrPooledNeighs,
        pOutGPUPtrNodeId, (int2*)pOutGPUPtrNeighs, pOutGPUPtrNStartIds);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### STORE POOLED NODES ###\n");
    fprintf(stderr, "Num nodes: %d\n", pNumNodes);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}