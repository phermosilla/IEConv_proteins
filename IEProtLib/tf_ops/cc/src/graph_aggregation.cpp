/////////////////////////////////////////////////////////////////////////////
/// \file graph_aggregation.cpp
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
#include "graph_aggregation.cuh"

/**
 *  Declaration of the tensorflow operation.
 */
REGISTER_OP("GraphAggregation")
    .Input("features: float32")
    .Input("neighbors: int32")
    .Input("start_indices: int32")
    .Output("aggr_features: float32")
    .Attr("normalize:int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(0));
        return Status::OK();
    });

REGISTER_OP("GraphAggregationGrads")
    .Input("in_grads: float32")
    .Input("neighbors: int32")
    .Input("start_indices: int32")
    .Output("out_grads: float32")
    .Attr("normalize:int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(0));
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to perform a graph aggregation operation.
     */
    class GraphAggregationOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit GraphAggregationOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                int auxNormalize;
                OP_REQUIRES_OK(pContext, pContext->GetAttr("normalize", &auxNormalize));
                normalize_ = auxNormalize!=0;
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inFeatures = pContext->input(0); 
                const Tensor& inNeighbors = pContext->input(1); 
                const Tensor& inStartIndices = pContext->input(2); 

                //Get variables from tensors.
                unsigned int numNodes = inFeatures.shape().dim_size(0);
                unsigned int numFeatures = inFeatures.shape().dim_size(1);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);

                //Get the pointers to GPU data from the tensors.
                const float* featuresGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inFeatures);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inStartIdsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inStartIndices);

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, inStartIndices.shape().dim_size(0) == numNodes, 
                    errors::InvalidArgument("GraphAggregationOp expects the same number of nodes"
                    " as start indices."));   
                OP_REQUIRES(pContext, inNeighbors.shape().dims() == 2 &&
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("GraphAggregationOp expects number of neighbors with the correct shape."));               

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numNodes, numFeatures};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the graph aggregation.
                mccnn::compute_graph_aggregation_gpu(gpuDevice, normalize_, numNodes, 
                    numFeatures, numNeighbors, featuresGPUPtr, inNeighborsGPUPtr, 
                    inStartIdsGPUPtr, outputGPUPtr);
            }
        
        private:

            /**Boolean that indicates if we need to normalize the aggregation.*/
            bool    normalize_;
    };

    /**
     *  Operation to perform a graph aggregation operation.
     */
    class GraphAggregationGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit GraphAggregationGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                int auxNormalize;
                OP_REQUIRES_OK(pContext, pContext->GetAttr("normalize", &auxNormalize));
                normalize_ = auxNormalize!=0;
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inGrads = pContext->input(0); 
                const Tensor& inNeighbors = pContext->input(1); 
                const Tensor& inStartIndices = pContext->input(2); 

                //Get variables from tensors.
                unsigned int numNodes = inGrads.shape().dim_size(0);
                unsigned int numFeatures = inGrads.shape().dim_size(1);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);

                //Get the pointers to GPU data from the tensors.
                const float* inGradsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGrads);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inStartIdsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inStartIndices);

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, inStartIndices.shape().dim_size(0) == numNodes, 
                    errors::InvalidArgument("GraphAggregationGradsOp expects the same number of nodes"
                    " as start indices."));   
                OP_REQUIRES(pContext, inNeighbors.shape().dims() == 2 &&
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("GraphAggregationGradsOp expects number of neighbors with the correct shape."));               

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numNodes, numFeatures};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the graph aggregation.
                mccnn::compute_graph_aggregation_grads_gpu(gpuDevice, normalize_, numNodes, 
                    numFeatures, numNeighbors, inGradsGPUPtr, inNeighborsGPUPtr, inStartIdsGPUPtr, outputGPUPtr);
            }
        
        private:

            /**Boolean that indicates if we need to normalize the aggregation.*/
            bool    normalize_;
    };
}
            
REGISTER_KERNEL_BUILDER(Name("GraphAggregation").Device(DEVICE_GPU), mccnn::GraphAggregationOp);
REGISTER_KERNEL_BUILDER(Name("GraphAggregationGrads").Device(DEVICE_GPU), mccnn::GraphAggregationGradsOp);