/////////////////////////////////////////////////////////////////////////////
/// \file compute_topo_dist.cu
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

#include "compute_topo_dist.cuh"

//With this configuration we can store the neighbors at distance 10.0 at the
//first atomic level of a protein.
#define NUM_SLOTS 128
#define NUM_ITERS 8

///////////////////////// GPU

template<int N>
__device__ unsigned int getIndexNeighbor(
    const unsigned int pNeighId,
    const unsigned int pNumSamples,
    const unsigned int pSavedIndexs,
    const int* __restrict__ pIds){
    
    unsigned int outIndex = N;
    bool found = false;
    for(int i = 0; i < pSavedIndexs && !found; ++i){
        int curId = pIds[i*pNumSamples];
        if(curId == pNeighId){
            outIndex = i;
            found = true;
        }
    }
    return outIndex;
}

/**
 *  GPU kernel to compute the neighbor list along the
 *  topology of a graph on the gpu.
 *  @param  pMaxDist                Maximum distance.
 *  @param  pNumSamples             Number of samples.
 *  @param  pPts                    Array of points.
 *  @param  pTopoNeighbors          Array of neighbors.
 *  @param  pTopoNeighIndexXSample  Indices of neighbors x sample.
 *  @param  pInNeighsIds            Input array with the current neighbors ids.
 *  @param  pInNeighsDist           Input array with the current neighbors distances.
 *  @param  pOutNeighsIds           Output array with the current neighbors ids.
 *  @param  pOutNeighsDist          Output array with the current neighbors distances.
 *  @paramt D                       Number of dimensions. 
 *  @paramt N                       Number of neighbor slots. 
 */
template<int D, int N>
__global__ void compute_topo_neighs_gpu_kernel(
    bool pConstEdge,
    const float pMaxDist,
    const unsigned int pNumSamples,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int2* __restrict__ pTopoNeighbors,
    const int* __restrict__ pTopoNeighIndexXSample,
    const int* __restrict__ pInNeighsIds,
    const float* __restrict__ pInNeighsDist,
    int* __restrict__ pOutNeighsIds,
    float* __restrict__ pOutNeighsDist)
{
    //Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the list of samples.
    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumSamples; 
        curIter += totalThreads)
    {
        //Get the current point coordinates.
        mccnn::fpoint<D> curPt = pPts[curIter];

        //Get the range of neighbors.
        int2 rangePts;
        rangePts.x = (curIter > 0)?pTopoNeighIndexXSample[curIter-1]:0;
        rangePts.y = pTopoNeighIndexXSample[curIter];
        int numPts = rangePts.y-rangePts.x;

        //Initialize the save index.
        int saveIndex = 0;

        //Iterate over the neighbors.
        for(int i = 0; i < numPts; ++i)
        {
            //Get the neighbor coordinate.
            int2 neighIndex = pTopoNeighbors[rangePts.x+i];

            //Add to the list.
            float neighDist = 1.0f;
            if(!pConstEdge){
                neighDist = mccnn::length(pPts[neighIndex.x] - curPt);
            }
            unsigned int curSaveIndex = getIndexNeighbor<N>(neighIndex.x, 
                pNumSamples, saveIndex, &pOutNeighsIds[curIter]);
            if(curSaveIndex < N){
                pOutNeighsDist[curIter+ curSaveIndex*pNumSamples] = 
                    MCCNN_MIN(pOutNeighsDist[curIter+ curSaveIndex*pNumSamples], neighDist);
            }else if(saveIndex < N){
                pOutNeighsIds[curIter + saveIndex*pNumSamples] = neighIndex.x;
                pOutNeighsDist[curIter+ saveIndex*pNumSamples] = neighDist;
                saveIndex++;
            }

            //Store the neighbors neighbors.
            bool endIter = false;
            for(int j = 0; j < N && !endIter; ++j)
            {
                //Get the list of neighbors of the neighbor.
                int curNeighId = pInNeighsIds[neighIndex.x + j*pNumSamples];
                float curNeighDist = pInNeighsDist[neighIndex.x + j*pNumSamples] + neighDist;

                //If it is a valid distance add to the list.
                if(curNeighId >= 0){
                    if(curNeighDist < pMaxDist){
                        unsigned int curSaveIndex = getIndexNeighbor<N>(
                            curNeighId, pNumSamples, saveIndex, &pOutNeighsIds[curIter]);
                        if(curSaveIndex < N){
                            pOutNeighsDist[curIter+ curSaveIndex*pNumSamples] = 
                                MCCNN_MIN(pOutNeighsDist[curIter+ curSaveIndex*pNumSamples], 
                                curNeighDist);
                        }else if(saveIndex < N){
                            pOutNeighsIds[curIter + saveIndex*pNumSamples] = curNeighId;
                            pOutNeighsDist[curIter + saveIndex*pNumSamples] = curNeighDist;
                            saveIndex++;
                        }
                    } 
                }else{
                    endIter = true;
                }             
            }
        }
    }
}

/**
 *  GPU kernel to select the distances along the topology
 *  for a given spatial neighborhood.
 *  @param  pMaxDist                Maximum distance.
 *  @param  pNumSamples             Number of samples.
 *  @param  pNeighbors              Array of neighbors.
 *  @param  pInNeighsIds            Input array with the current neighbors ids.
 *  @param  pInNeighsDist           Input array with the current neighbors distances.
 *  @param  pOutNeighsDist          Output array with the current neighbors distances.
 *  @paramt D                       Number of dimensions. 
 *  @paramt N                       Number of neighbor slots. 
 */
 template<int D, int N>
 __global__ void select_topo_dists_gpu_kernel(
     const float pMaxDist,
     const unsigned int pNumSamples,
     const unsigned int pNumNeighbors,
     const int2* __restrict__ pNeighbors,
     const int* __restrict__ pInNeighsIds,
     const float* __restrict__ pInNeighsDist,
     float* __restrict__ pOutNeighsDist)
{
    //Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Iterate over the list of samples.
    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumNeighbors; 
        curIter += totalThreads)
    {
        //Get the current neighbor.
        int2 curNeigh = pNeighbors[curIter];

        float distance = pMaxDist;
        if(curNeigh.x == curNeigh.y){
            distance = 0.0f;
        }else{
            bool endIter = false;
            for(int j = 0; j < N && !endIter; ++j)
            {
                //Get the list of neighbors of the neighbor.
                int curNeighDistId = pInNeighsIds[curNeigh.y + j*pNumSamples];

                //Check if it is the neighbor we are interested in.
                if(curNeighDistId >= 0){
                    if(curNeighDistId == curNeigh.x){
                        distance = pInNeighsDist[curNeigh.y + j*pNumSamples];
                    }
                }else{
                    endIter = true;
                }
            }
        }

        //Save the distance.
        pOutNeighsDist[curIter] = distance;
    }
}

///////////////////////// CPU

template<int D>
void mccnn::compute_topo_dist_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    bool pConstEdge,
    float pMaxDist,
    const unsigned int pNumSamples,
    const unsigned int pNumSpatialNeighbors,
    const float* pInGPUPtrPts,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrTopo,
    const int* pInGPUPtrSampleTopoI,
    float* pOutGPUPtrDists)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

#ifdef DEBUG_INFO
    cudaEvent_t start, stop, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stop2);
    cudaEventRecord(start, cudaStream);
#endif

    //Store the temporal buffers.
    int* tmpBufferIds1 = pDevice->getIntTmpGPUBuffer(pNumSamples*NUM_SLOTS);
    int* tmpBufferIds2 = pDevice->getIntTmpGPUBuffer(pNumSamples*NUM_SLOTS);
    float* tmpBufferDists1 = pDevice->getFloatTmpGPUBuffer(pNumSamples*NUM_SLOTS);
    float* tmpBufferDists2 = pDevice->getFloatTmpGPUBuffer(pNumSamples*NUM_SLOTS);

    //Initialize the temporal buffers.
    pDevice->memset(tmpBufferIds1, -1, sizeof(int)*pNumSamples*NUM_SLOTS);
    pDevice->memset(tmpBufferIds2, -1, sizeof(int)*pNumSamples*NUM_SLOTS);
    pDevice->memset(tmpBufferDists1, 0, sizeof(float)*pNumSamples*NUM_SLOTS);
    pDevice->memset(tmpBufferDists2, 0, sizeof(float)*pNumSamples*NUM_SLOTS);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)compute_topo_neighs_gpu_kernel<D, NUM_SLOTS>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumSamples/blockSize;
    execBlocks += (pNumSamples%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernels to compute the neighbors at a certain distance along
    //the topology of the graph.
    int* auxBufferIds1 = tmpBufferIds1;
    int* auxBufferIds2 = tmpBufferIds2;
    float* auxBufferDist1 = tmpBufferDists1;
    float* auxBufferDist2 = tmpBufferDists2;
    for(int iter = 0; iter < NUM_ITERS; ++iter){
        compute_topo_neighs_gpu_kernel<D, NUM_SLOTS><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pConstEdge, pMaxDist, pNumSamples, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts,
            (const int2*)pInGPUPtrTopo,
            pInGPUPtrSampleTopoI,
            auxBufferIds1,
            auxBufferDist1,
            auxBufferIds2,
            auxBufferDist2);
        pDevice->check_error(__FILE__, __LINE__);
        int* auxTmpVar1 = auxBufferIds1;
        float* auxTmpVar2 = auxBufferDist1;
        auxBufferIds1 = auxBufferIds2;
        auxBufferDist1 = auxBufferDist2;
        auxBufferIds2 = auxTmpVar1;
        auxBufferDist2 = auxTmpVar2;
    }

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### COMPUTE TOPO NEIGHS 1 ###\n");
    fprintf(stderr, "Num samples: %d\n", pNumSamples);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numBlocks2 = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)select_topo_dists_gpu_kernel<D, NUM_SLOTS>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks2 = pNumSpatialNeighbors/blockSize;
    execBlocks2 += (pNumSpatialNeighbors%blockSize != 0)?1:0;
    unsigned int totalNumBlocks2 = numMP*numBlocks2;
    totalNumBlocks2 = (totalNumBlocks2 > execBlocks2)?execBlocks2:totalNumBlocks2;

    //Save the final distances.
    select_topo_dists_gpu_kernel<D, NUM_SLOTS><<<totalNumBlocks2, blockSize, 0, cudaStream>>>(
        pMaxDist, pNumSamples, pNumSpatialNeighbors,
        (const int2*)pInGPUPtrNeighbors,
        auxBufferIds1, auxBufferDist1, pOutGPUPtrDists);
    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop2, cudaStream);
    cudaEventSynchronize(stop2);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, stop, stop2);

    gpuOccupancy = (float)(numBlocks2*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### COMPUTE TOPO NEIGHS 2 ###\n");
    fprintf(stderr, "Num neighbors: %d\n", pNumSpatialNeighbors);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif

}

///////////////////////// CPU Template declaration

#define COMPUTE_TOPO_NEIGHS_TEMP_DECL(Dims)             \
    template void mccnn::compute_topo_dist_gpu<Dims>(   \
        std::unique_ptr<IGPUDevice>& pDevice,           \
        bool pConstEdge,                                \
        float pMaxDist,                                 \
        const unsigned int pNumSamples,                 \
        const unsigned int pNumSpatialNeighbors,        \
        const float* pInGPUPtrPts,                      \
        const int* pInGPUPtrNeighbors,                  \
        const int* pInGPUPtrTopo,                       \
        const int* pInGPUPtrSampleTopoI,                \
        float* pOutGPUPtrDists);

DECLARE_TEMPLATE_DIMS(COMPUTE_TOPO_NEIGHS_TEMP_DECL)