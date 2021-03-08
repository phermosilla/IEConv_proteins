/////////////////////////////////////////////////////////////////////////////
/// \file basis_hproj_bilateral.cuh
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
#include "cuda_kernel_utils.cuh"
#include "math_helper.cuh"
#include "nn_utils.cuh"
#include "basis/basis_hproj_bilateral.cuh"
#include "basis/basis_utils.cuh"

template<int D, int K, int U, int A>
__global__ void compute_hproj_bilateral_basis_proj_pt_coords(
    const unsigned int pNumNeighbors,       
    const mccnn::fpoint<D>* __restrict__ pInPtsGPUPtr,
    const mccnn::fpoint<D>* __restrict__ pInSamplesGPUPtr,
    const mccnn::fpoint<D>* __restrict__ pInInvRadiiGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInXNeighValsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    float* __restrict__ pOutProjGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float kernelPts[];

    //Create the struct to compute the activation function.
    mccnn::activation_function_struct<A> acFunc;

    //Load the kernel point centers.
#pragma unroll(2)
    for(int i = threadIdx.x; i < K*(D+U+1); i+=blockDim.x)
        kernelPts[i] = pInBasisGPUPtr[i];

    __syncthreads();

    //Get usefull indices.
    const unsigned int initThreadIndex = mccnn::compute_global_index_gpu_funct();
    const unsigned int totalNumThreads = mccnn::compute_total_threads_gpu_funct(); 

    for(unsigned int curIter = initThreadIndex; 
        curIter < pNumNeighbors; curIter += totalNumThreads)
    {
        //Get indices to the point and sample.
        int2 neighAndSampleIndices = pInNeighborsGPUPtr[curIter];

        //Compute the pt difference.
        mccnn::fpoint<D> ptDiff = (pInPtsGPUPtr[neighAndSampleIndices.x] - 
            pInSamplesGPUPtr[neighAndSampleIndices.y])*pInInvRadiiGPUPtr[0];

        //Compute the pdf inverse.                
        float weightVal = 1.0f/(pInPDFsGPUPtr[curIter]);

        //Compute the projection of each basis.
        for(int i = 0; i < K; ++i){
            float sum = 0.0f;
#pragma unroll
            for(int j = 0; j < D; ++j)
                sum += kernelPts[i*(D+U+1) + j]*ptDiff[j];
#pragma unroll
            for(int j = 0; j < U; ++j)
                sum += kernelPts[i*(D+U+1) + D + j]*
                    pInXNeighValsGPUPtr[curIter*U + j];
            sum += kernelPts[i*(D+U+1) + D + U];
            pOutProjGPUPtr[curIter*K + i] = acFunc.forward(sum)*weightVal;
        }
    }
}

/**
 *  Template to accumulate the point gradients.
 */
 template<int D, int K, int U, bool P> 
 struct accum_pt_grads{
 
     __forceinline__ __device__ void accumulate(
         const int pOffset,
         const float* pSharedMem,
         float* __restrict__ pOutPtGrads,
         float* __restrict__ pOutSampleGrads,
         float* __restrict__ pOutPDFGrads,
         float* __restrict__ pXNeighValGrads){}
 };
 
 template<int D, int K, int U> 
 struct accum_pt_grads<D, K, U, true>{
 
     __forceinline__ __device__ void accumulate(
         const int pOffset,
         const float* __restrict__ pSharedMem,
         float* __restrict__ pOutPtGrads,
         float* __restrict__ pOutSampleGrads,
         float* __restrict__ pOutPDFGrads,
         float* __restrict__ pXNeighValGrads){
         float accumVal = 0.0f;
 #pragma unroll
        for(int j = 0; j < K; ++j){
            accumVal += pSharedMem[pOffset*blockDim.x + j];
        }
        if(pOffset < D)
            atomicAdd(&pOutPtGrads[pOffset], accumVal);
        else if(pOffset < D+U)
            atomicAdd(&pXNeighValGrads[pOffset - D], accumVal);
        else if(pOffset < (D*2+U))
            atomicAdd(&pOutSampleGrads[pOffset - (D+U)], accumVal);
        else
            pOutPDFGrads[0] = accumVal;
     }
 };

template<int D, int K, int U, int A, bool P>
__global__ void compute_hproj_bilateral_basis_proj_pt_coords_grads(
    const unsigned int pNumNeighbors,       
    const float* __restrict__ pInPtsGPUPtr,
    const float* __restrict__ pInSamplesGPUPtr,
    const float* __restrict__ pInInvRadiiGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInXNeighValsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    const float* __restrict__ pInGradsGPUPtr,
    float* __restrict__ pOutBasisGradsGPUPtr,
    float* __restrict__ pOutPtsGradsGPUPtr,
    float* __restrict__ pOutSampleGradsGPUPtr,
    float* __restrict__ pOutPDFGradsGPUPtr,
    float* __restrict__ pOutXNeighValsGradsGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float sharedMem[];

    //Create the struct to compute the activation function.
    mccnn::activation_function_struct<A> acFunc;

    //Create the struct to compute point gradients.
    accum_pt_grads<D, K, U, P> ptGrads;

    //Compute usefull indices.
    int totalExecThreads = pNumNeighbors*K;
    totalExecThreads += (totalExecThreads%blockDim.x != 0)?
        blockDim.x-totalExecThreads%blockDim.x:0;
    int groupId = threadIdx.x/K;
    int kpIndex = threadIdx.x%K;
    int groupsXBlock = blockDim.x/K;

    //Get the pointers to shared memory.
    float* kernelPts = sharedMem;
    float* accumGrads = &sharedMem[K*(D+U+1)];
    float* sharedPtDiffs = &sharedMem[K*(D+U+1) + blockDim.x*(D+U+1)];
    float* accumPtGrads = &sharedMem[K*(D+U+1) + blockDim.x*(D+U+1) + groupsXBlock*(D+U)];

    //Load the kernel point centers.
#pragma unroll(2)
    for(int i = threadIdx.x; i < K*(D+U+1); i+=blockDim.x)
        kernelPts[i] = pInBasisGPUPtr[i];

#pragma unroll
    for(int i = 0; i < D+U+1; ++i)
        accumGrads[i*blockDim.x + threadIdx.x] = 0.0f;

    //Get usefull indices.
    const int initThreadIndex = mccnn::compute_global_index_gpu_funct();
    const int totalNumThreads = mccnn::compute_total_threads_gpu_funct(); 

    for(int curIter = initThreadIndex; 
        curIter < totalExecThreads; 
        curIter += totalNumThreads)
    {
        //Get indices to the point and sample.
        int2 neighAndSampleIndices;
        int neighIndex = curIter/K;
        float inGradient = 0.0f;

        if(neighIndex < pNumNeighbors){
            neighAndSampleIndices = pInNeighborsGPUPtr[neighIndex];

            //Compute the pt difference.
            if(kpIndex < D){
                sharedPtDiffs[groupId*(D+U) + kpIndex] = 
                    (pInPtsGPUPtr[neighAndSampleIndices.x*D + kpIndex] -
                    pInSamplesGPUPtr[neighAndSampleIndices.y*D + kpIndex])*
                    pInInvRadiiGPUPtr[kpIndex];
            }else if(kpIndex < D+U){
                sharedPtDiffs[groupId*(D+U) + kpIndex] = 
                    pInXNeighValsGPUPtr[neighIndex*U + kpIndex - D];
            }

            //Get the gradient.
            inGradient = pInGradsGPUPtr[neighIndex*K + kpIndex];
        }

        __syncthreads();

        if(neighIndex < pNumNeighbors){
            //Compute the pdf inverse.                
            float invPdf = 1.0f/(pInPDFsGPUPtr[neighIndex]);

            //Compute the projection of each basis.
            float sum = 0.0f;
#pragma unroll
            for(int j = 0; j < D+U; ++j)
                sum += kernelPts[kpIndex*(D+U+1) + j]*sharedPtDiffs[groupId*(D+U) + j];
            sum += kernelPts[kpIndex*(D+U+1) + D + U];
            float value = acFunc.forward(sum);

            //Compute the gradient before the projection.
            float curInGradient = inGradient * acFunc.backward(value) * invPdf;

            //Compute the gradients
            //TODO - Add kahan summation, but requires more shared memory.
#pragma unroll
            for(int j = 0; j < D+U; ++j){
                accumGrads[threadIdx.x + j*blockDim.x] += 
                    sharedPtDiffs[groupId*(D+U) + j]*curInGradient;
                if (j < D){
                    accumPtGrads[threadIdx.x + j*blockDim.x] = 
                        pInInvRadiiGPUPtr[j]*curInGradient*kernelPts[kpIndex*(D+U+1) + j];
                    accumPtGrads[threadIdx.x + (D+U+j)*blockDim.x] = 
                        -pInInvRadiiGPUPtr[j]*curInGradient*kernelPts[kpIndex*(D+U+1) + j];
                }else{
                    accumPtGrads[threadIdx.x + j*blockDim.x] = curInGradient*
                        kernelPts[kpIndex*(D+U+1) + j];
                }
            }
            accumGrads[threadIdx.x + (D+U)*blockDim.x] += curInGradient;//Bias
            accumPtGrads[threadIdx.x + (D*2+U)*blockDim.x] = -value*invPdf*invPdf*inGradient;//PDF
        }

        __syncthreads();

        if(neighIndex < pNumNeighbors && kpIndex < (D*2+U+1)){
            ptGrads.accumulate(kpIndex, &accumPtGrads[groupId*K],
                &pOutPtsGradsGPUPtr[neighAndSampleIndices.x*D],
                &pOutSampleGradsGPUPtr[neighAndSampleIndices.y*D],
                &pOutPDFGradsGPUPtr[neighIndex], 
                &pOutXNeighValsGradsGPUPtr[neighIndex*U]);
        }

        __syncthreads();
    }

    //Save the gradient into memory.
    for(int i = threadIdx.x; i < K*(D+U+1); i+=blockDim.x){
        int dimension = i/K;
        int kpoint = i%K;
        float accumVal = 0.0f;
#pragma unroll(2)
        for(int j = 0; j < groupsXBlock; ++j){
            accumVal += accumGrads[dimension*blockDim.x + j*K + kpoint];
        }
        atomicAdd(&pOutBasisGradsGPUPtr[kpoint*(D+U+1) + dimension], accumVal);
        /*if(initThreadIndex < 64 && kpoint < 2){
            printf("%f %d %d | ", accumVal, kpoint, dimension);
        }*/
    }
}

/////////////////// CLASS DEFINITION

namespace mccnn{
        
    template<int D, int K, int U>
    HProjBilateralBasis<D, K, U>::HProjBilateralBasis(
        HProjBilateralBasis::ActivationFunction pAcFunc)
        :BasisInterface<D, K, U>(), acFunc_(pAcFunc)
    {
    }

    template<int D, int K, int U>
    HProjBilateralBasis<D, K, U>::~HProjBilateralBasis(void)
    {
    }

    template<int D, int K, int U>
    void HProjBilateralBasis<D, K, U>::compute_basis_proj_pt_coords(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNeighbors,       
        const float* pInPtsGPUPtr,
        const float* pInSamplesGPUPtr,
        const float* pInInvRadiiGPUPtr,
        const int* pInNeighborsGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInXNeighValsGPUPtr,
        const float* pInBasisGPUPtr,
        float* pOutProjGPUPtr)
    {
        //Get the device properties.
        const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

        //Get information of the Device.
        unsigned int numMP = gpuProps.numMPs_;

        //Get the cuda stream.
        auto cudaStream = pDevice->getCUDAStream();

        //Define the block size.
        unsigned int blockSize = 64;

        //Get the current function pointer.
        const void* cFunct = nullptr;
        if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::RELU){
            cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 0>;
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::LRELU){
            cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 1>;
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::ELU){
            cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 2>;
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::EXP){
            cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 3>;
        }

#ifdef DEBUG_INFO
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, cudaStream);
#endif
        //Calculate the shared memory needed.
        unsigned int sharedMemSize = (K*(D+U+1)*sizeof(float));

        //Compute the number of blocks
        unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
            blockSize, cFunct, sharedMemSize);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = pNumNeighbors/blockSize;
        execBlocks += (pNumNeighbors%blockSize != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;
        
        //Execute the kernel extensions.
        if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::RELU){
            compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 0>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, 
                (const fpoint<D>*)pInPtsGPUPtr,
                (const fpoint<D>*)pInSamplesGPUPtr,
                (const fpoint<D>*)pInInvRadiiGPUPtr,
                (const int2*)pInNeighborsGPUPtr,
                pInPDFsGPUPtr, pInXNeighValsGPUPtr, 
                pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::LRELU){
            compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 1>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, 
                (const fpoint<D>*)pInPtsGPUPtr,
                (const fpoint<D>*)pInSamplesGPUPtr,
                (const fpoint<D>*)pInInvRadiiGPUPtr,
                (const int2*)pInNeighborsGPUPtr,
                pInPDFsGPUPtr, pInXNeighValsGPUPtr, 
                pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::ELU){
            compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 2>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, 
                (const fpoint<D>*)pInPtsGPUPtr,
                (const fpoint<D>*)pInSamplesGPUPtr,
                (const fpoint<D>*)pInInvRadiiGPUPtr,
                (const int2*)pInNeighborsGPUPtr,
                pInPDFsGPUPtr, pInXNeighValsGPUPtr, 
                pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::EXP){
            compute_hproj_bilateral_basis_proj_pt_coords<D, K, U, 3>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, 
                (const fpoint<D>*)pInPtsGPUPtr,
                (const fpoint<D>*)pInSamplesGPUPtr,
                (const fpoint<D>*)pInInvRadiiGPUPtr,
                (const int2*)pInNeighborsGPUPtr,
                pInPDFsGPUPtr, pInXNeighValsGPUPtr,
                pInBasisGPUPtr, pOutProjGPUPtr);
        }
        pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
        cudaEventRecord(stop, cudaStream);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        struct cudaFuncAttributes funcAttrib;
        cudaFuncGetAttributes(&funcAttrib, cFunct);
        float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

        fprintf(stderr, "### HPROJ BILATERAL BASIS PROJ ###\n");
        fprintf(stderr, "Num basis: %d\n", K);
        fprintf(stderr, "Local memory: %d\n", (int)funcAttrib.localSizeBytes);
        fprintf(stderr, "Constant memory: %d\n", (int)funcAttrib.constSizeBytes);
        fprintf(stderr, "Num reg kernel: %d\n", funcAttrib.numRegs);
        fprintf(stderr, "Shared memory kernel: %d\n", sharedMemSize);
        fprintf(stderr, "Num neighbors: %d\n", pNumNeighbors);
        fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
        fprintf(stderr, "Execution time: %f\n", milliseconds);
        fprintf(stderr, "\n");
#endif
    }

    template<int D, int K, int U>
    void HProjBilateralBasis<D, K, U>::compute_grads_basis_proj_pt_coords(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNeighbors,       
        const float* pInPtsGPUPtr,
        const float* pInSamplesGPUPtr,
        const float* pInInvRadiiGPUPtr,
        const int* pInNeighborsGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInXNeighValsGPUPtr,
        const float* pInBasisGPUPtr,
        const float* pInGradsGPUPtr,
        float* pOutBasisGradsGPUPtr,
        float* pOutPtsGradsGPUPtr,
        float* pOutSampleGradsGPUPtr,
        float* pOutPDFGradsGPUPtr,
        float* pOutXNeighGradsGPUPtr)
    {
        //Check if the gradietns of the points should be computed.
        bool pointGrads = (pOutPtsGradsGPUPtr != nullptr) &&
            (pOutSampleGradsGPUPtr != nullptr) &&
            (pOutPDFGradsGPUPtr != nullptr);
        
        //Get the device properties.
        const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

        //Get information of the Device.
        unsigned int numMP = gpuProps.numMPs_;

        //Get the cuda stream.
        auto cudaStream = pDevice->getCUDAStream();

        //Define the block size.
        unsigned int blockSize = 64;

        //Get the current function pointer.
        const void* cFunct = nullptr;
        if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::RELU){
            if(pointGrads)
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 0, true>;
            else
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 0, false>;
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::LRELU){
            if(pointGrads)
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 1, true>;
            else
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 1, false>;
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::ELU){
            if(pointGrads)
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 2, true>;
            else
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 2, false>;
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::EXP){
            if(pointGrads)
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 3, true>;
            else
                cFunct = (const void*)compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 3, false>;
        }

#ifdef DEBUG_INFO
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, cudaStream);
#endif

        //Calculate the shared memory needed.
        unsigned int sharedMemSize = ((K + blockSize)*(D+U+1) + 
            (blockSize/K)*(D+U) + blockSize*(D*2+ U + 1))*sizeof(float);

        //Compute the number of blocks
        unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
            blockSize, cFunct, sharedMemSize);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = (pNumNeighbors*K)/blockSize;
        execBlocks += ((pNumNeighbors*K)%blockSize != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

        //Execute the kernel extensions.
        if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::RELU){
            if(pointGrads){
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 0, true>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }else{
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 0, false>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::LRELU){
            if(pointGrads){
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 1, true>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }else{
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 1, false>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::ELU){
            if(pointGrads){
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 2, true>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }else{
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 2, false>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }
        }else if(acFunc_ == HProjBilateralBasis<D, K, U>::ActivationFunction::EXP){
            if(pointGrads){
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 3, true>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }else{
                compute_hproj_bilateral_basis_proj_pt_coords_grads<D, K, U, 3, false>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInXNeighValsGPUPtr, pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr,
                    pOutXNeighGradsGPUPtr);
            }
        }
        
        pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
        cudaEventRecord(stop, cudaStream);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        struct cudaFuncAttributes funcAttrib;
        cudaFuncGetAttributes(&funcAttrib, cFunct);
        float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

        fprintf(stderr, "### HPROJ BILATERAL BASIS PROJ GRADS ###\n");
        fprintf(stderr, "Num basis: %d\n", K);
        fprintf(stderr, "Local memory: %d\n", (int)funcAttrib.localSizeBytes);
        fprintf(stderr, "Constant memory: %d\n", (int)funcAttrib.constSizeBytes);
        fprintf(stderr, "Num reg kernel: %d\n", funcAttrib.numRegs);
        fprintf(stderr, "Shared memory kernel: %d\n", sharedMemSize);
        fprintf(stderr, "Num neighbors: %d\n", pNumNeighbors);
        fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
        fprintf(stderr, "Execution time: %f\n", milliseconds);
        fprintf(stderr, "\n");
#endif
    }
}

//DECLARE THE VALID INSTANCES OF THE TEMPLATE CLASS
#define HPROJ_BILATERAL_BASIS_CLASS_DECL(D, K, U)    \
template class mccnn::HProjBilateralBasis<D, K, U>;
DECLARE_TEMPLATE_DIMS_BASIS(HPROJ_BILATERAL_BASIS_CLASS_DECL)