/////////////////////////////////////////////////////////////////////////////
/// \file compute_topo_dist.cpp
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
#include "compute_topo_dist.cuh"

/**
 *  Declaration of the tensorflow operations.
 */
REGISTER_OP("ComputeTopoDist")
    .Input("points: float32")
    .Input("neighbors: int32")
    .Input("topology: int32")
    .Input("sample_topo_start_indices: int32")
    .Output("dists: float32")
    .Attr("max_dist: float")
    .Attr("const_edge: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({pIC->Dim(pIC->input(1), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to compute the distance along the graph
     *  topology.
     */
    class ComputeTopoDistOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit ComputeTopoDistOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                OP_REQUIRES_OK(pContext, pContext->GetAttr("max_dist", &maxDist_));
                OP_REQUIRES(pContext, maxDist_ > 0.0, 
                    errors::InvalidArgument("ComputeTopoDist requires positive max distance."));

                int constEdge;
                OP_REQUIRES_OK(pContext, pContext->GetAttr("const_edge", &constEdge));
                OP_REQUIRES(pContext, constEdge == 0 || constEdge == 1, 
                    errors::InvalidArgument("ComputeTopoDist requires const_edge equal to 1 or 0."));
                constEdge_ = constEdge == 1;
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0); 
                const Tensor& inNeighbors = pContext->input(1); 
                const Tensor& inTopology = pContext->input(2); 
                const Tensor& inSampleTopoStartIndices = pContext->input(3);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSampleTopoStartIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numTopoNeighs = inTopology.shape().dim_size(0);

                //Get the pointers to GPU data from the tensors.
                const float* ptsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const int* neighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* topoGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inTopology);
                const int* sampleTopoIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleTopoStartIndices);
                
                //Check for the correctness of the input.
                if(!constEdge_){
                    OP_REQUIRES(pContext, numSamples == numPts, 
                        errors::InvalidArgument("ComputeTopoDist expects the same points as samples."));
                }
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("ComputeTopoDist expects a valid number of dimension"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("ComputeTopoDist expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numNeighbors};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the distances along the topology.
                DIMENSION_SWITCH_CALL(numDimensions, mccnn::compute_topo_dist_gpu,
                    gpuDevice, constEdge_, maxDist_, numSamples, numNeighbors, ptsGPUPtr,
                    neighborsGPUPtr, topoGPUPtr, sampleTopoIGPUPtr, outputGPUPtr);
            }

        private:

            /**Maximum distance.*/
            float   maxDist_;
            /**Boolean that indicates if we use a constant value for the edge.*/
            bool    constEdge_;
    };
}

REGISTER_KERNEL_BUILDER(Name("ComputeTopoDist").Device(DEVICE_GPU), mccnn::ComputeTopoDistOp);
