/////////////////////////////////////////////////////////////////////////////
/// \file protein_pooling.cpp
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
#include "tf_utils.hpp"
#include "tf_gpu_device.hpp"
#include "protein_pooling.cuh"
#include "scan_alg.cuh"

/**
 *  Declaration of the tensorflow operation.
 */
REGISTER_OP("ProteinPooling")
    .Input("neighbors: int32")
    .Input("start_indices: int32")
    .Output("pooled_indices: int32")
    .Output("pooled_neighs: int32")
    .Output("pooled_start_ids: int32")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims0 = pIC->MakeShape({-1});
        shape_inference::ShapeHandle outputDims1 = pIC->MakeShape({-1, 2});        
        pIC->set_output(0, outputDims0);
        pIC->set_output(1, outputDims1);
        pIC->set_output(2, outputDims0);
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to perform a graph pooling operation.
     */
    class ProteinPoolingOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit ProteinPoolingOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){}
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inNeighbors = pContext->input(0); 
                const Tensor& inStartIndices = pContext->input(1); 

                //Get variables from tensors.
                unsigned int numNodes = inStartIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);

                //Get the pointers to GPU data from the tensors.
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inStartIdsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inStartIndices);

                //Check for the correctness of the input.  
                OP_REQUIRES(pContext, inNeighbors.shape().dims() == 2 &&
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("ProteinPoolingOp expects number of neighbors with the correct shape."));               

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create a temporal tensor.
                int* tmpGPUPtr = gpuDevice->getIntTmpGPUBuffer(numNodes);

                //Compute the graph aggregation.
                mccnn::compute_start_backbones_gpu(gpuDevice, numNodes, 
                    inNeighborsGPUPtr, inStartIdsGPUPtr, tmpGPUPtr);

                //Scan algorithm.
                unsigned int numBackBones =  mccnn::scan_alg(gpuDevice, numNodes, tmpGPUPtr);

                //Create a temporal tensor.
                int* tmpGPUPtr2 = gpuDevice->getIntTmpGPUBuffer(numBackBones);

                //Store the starting indices for each backbone.
                mccnn::store_start_backbones_gpu(gpuDevice, numNodes, inNeighborsGPUPtr,
                    inStartIdsGPUPtr, tmpGPUPtr, tmpGPUPtr2);

                //Create a temporal tensor.
                int* tmpGPUPtr3 = gpuDevice->getIntTmpGPUBuffer(numNodes);
                int* tmpGPUPtr4 = gpuDevice->getIntTmpGPUBuffer(numNodes);

                //Counte the number of pooled nodes.
                mccnn::count_pooled_nodes_gpu(gpuDevice, numNodes, inNeighborsGPUPtr, 
                    inStartIdsGPUPtr, tmpGPUPtr2, tmpGPUPtr, tmpGPUPtr3, tmpGPUPtr4);

                //Scan algorithm.
                unsigned int numPooledNodes =  mccnn::scan_alg(gpuDevice, numNodes, tmpGPUPtr3);
                unsigned int numPooledNeighs =  mccnn::scan_alg(gpuDevice, numNodes, tmpGPUPtr4);

                //Create the output tensors.
                int* output1GPUPtr = nullptr;
                int* output2GPUPtr = nullptr;
                int* output3GPUPtr = nullptr;
                TensorShape outShape1 = TensorShape{numPooledNodes};
                TensorShape outShape2 = TensorShape{numPooledNeighs, 2};
                TensorShape outShape3 = TensorShape{numPooledNodes};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                    (0, pContext, outShape1, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                    (1, pContext, outShape2, &output2GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                    (2, pContext, outShape3, &output3GPUPtr));

                //Fill the output vectors.
                mccnn::pool_nodes_gpu(gpuDevice, numNodes, 
                    numPooledNodes, numPooledNeighs,
                    inNeighborsGPUPtr, inStartIdsGPUPtr,
                    tmpGPUPtr3, tmpGPUPtr4, output1GPUPtr,
                    output2GPUPtr, output3GPUPtr);

            }
    };
}
            
REGISTER_KERNEL_BUILDER(Name("ProteinPooling").Device(DEVICE_GPU), mccnn::ProteinPoolingOp);