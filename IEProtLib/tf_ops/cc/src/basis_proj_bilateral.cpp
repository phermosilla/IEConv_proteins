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
#include "tf_utils.hpp"
#include "tf_gpu_device.hpp"
#include "basis/basis_utils.cuh"
#include "basis/basis_proj.cuh"
#include "basis/basis_proj_grads.cuh"

/**
 *  Declaration of the tensorflow operations.
 */
REGISTER_OP("BasisProjBil")
    .Input("points: float32")
    .Input("pt_features: float32")
    .Input("samples: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_radii: float32")
    .Input("pdfs: float32")
    .Input("neigh_vals: float32")
    .Input("basis_func: float32")
    .Output("features: float32")
    .Attr("basis_type: int")
    .Attr("pt_grads: bool")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({
                pIC->Dim(pIC->input(2), 0), 
                pIC->Dim(pIC->input(1), 1),
                pIC->Dim(pIC->input(7), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("BasisProjBilGrads")
    .Input("points: float32")
    .Input("pt_features: float32")
    .Input("samples: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_radii: float32")
    .Input("pdfs: float32")
    .Input("neigh_vals: float32")
    .Input("basis_func: float32")
    .Input("in_gradietns: float32")
    .Output("feat_gradients: float32")
    .Output("basis_gradients: float32")
    .Attr("basis_type: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(1));
        pIC->set_output(1, pIC->input(8));
        return Status::OK();
    });

REGISTER_OP("BasisProjBilGradsWithPtGrads")
    .Input("points: float32")
    .Input("pt_features: float32")
    .Input("samples: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_radii: float32")
    .Input("pdfs: float32")
    .Input("neigh_vals: float32")
    .Input("basis_func: float32")
    .Input("in_gradietns: float32")
    .Output("feat_gradients: float32")
    .Output("basis_gradients: float32")
    .Output("point_gradients: float32")
    .Output("sample_gradients: float32")
    .Output("pdf_gradients: float32")
    .Output("neigh_vals_gradients: float32")
    .Attr("basis_type: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(1));
        pIC->set_output(1, pIC->input(8));
        pIC->set_output(2, pIC->input(0));
        pIC->set_output(3, pIC->input(2));
        pIC->set_output(4, pIC->input(6));
        pIC->set_output(5, pIC->input(7));
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to project input features into a set of basis functions
     *  for a bilateral filter.
     */
    class BasisProjBilOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjBilOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("basis_type", &basisType_));
                OP_REQUIRES(pContext, basisType_ >= 6 && basisType_ < 10, 
                    errors::InvalidArgument("BasisProjBilOp requires a valid basis type."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inSamples = pContext->input(2); 
                const Tensor& inNeighbors = pContext->input(3); 
                const Tensor& inSampleNeighIndices = pContext->input(4);
                const Tensor& inInvRadii = pContext->input(5);
                const Tensor& inPDFs = pContext->input(6);
                const Tensor& inNeighVals = pContext->input(7);
                const Tensor& inBasis = pContext->input(8);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);
                unsigned int numNeighVals = inNeighVals.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inPtsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const float* inSamplesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSamples);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inNeighValsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inNeighVals);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions, numNeighVals);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjBilOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BasisProjBilOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjBilOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inSamples.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("BasisProjBilOp expects a number of dimensions in"
                    " inSamples equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BasisProjBilOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inPtFeatures.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("BasisProjBilOp expects a number of points in"
                    " inPtFeatures equal to the number of points in the input points tensor"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjBilOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inSampleNeighIndices.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("BasisProjBilOp expects the same number of samples "
                    "in inSampleNeighIndices as in the samples tensor."));
                OP_REQUIRES(pContext, inPDFs.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjBilOp expects a number of pdf values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inNeighVals.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjBilOp expects a number of neighbor values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inBasis.dims() == 2 && 
                    inBasis.shape().dim_size(1) == numParamsBasis, 
                    errors::InvalidArgument("BasisProjBilOp expects the rigth number of "
                    "parameters each for each basis function."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numSamples, numInFeatures, numBasis};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the convolution.
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, numNeighVals, 
                    mccnn::basis_proj_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, numSamples, numNeighbors, 
                    numInFeatures, inPtsGPUPtr, inPtFeaturesGPUPtr, inSamplesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inInvRadiiGPUPtr, 
                    inBasisGPUPtr, inPDFsGPUPtr, inNeighValsGPUPtr, outputGPUPtr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };


    /**
     *  Operation to compute a monte carlo convolution.
     */
    class BasisProjBilGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjBilGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("basis_type", &basisType_));
                OP_REQUIRES(pContext, basisType_ >= 6 && basisType_ < 10, 
                    errors::InvalidArgument("BasisProjBilGradsOp requires a valid basis type."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inSamples = pContext->input(2); 
                const Tensor& inNeighbors = pContext->input(3); 
                const Tensor& inSampleNeighIndices = pContext->input(4);
                const Tensor& inInvRadii = pContext->input(5);
                const Tensor& inPDFs = pContext->input(6);
                const Tensor& inNeighVals = pContext->input(7);
                const Tensor& inBasis = pContext->input(8);
                const Tensor& inGradients = pContext->input(9);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);
                unsigned int numNeighVals = inNeighVals.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inPtsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const float* inSamplesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSamples);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inNeighValsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inNeighVals);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);
                const float* inGradientsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGradients);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions, numNeighVals);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inSamples.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a number of dimensions in"
                    " inSamples equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inPtFeatures.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a number of points in"
                    " inPtFeatures equal to the number of points in the input points tensor"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inSampleNeighIndices.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects the same number of points "
                    "in inSampleNeighIndices as in the samples tensor."));
                OP_REQUIRES(pContext, inPDFs.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a number of pdf values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inNeighVals.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects a number of neighbor values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inBasis.dims() == 2 && 
                    inBasis.shape().dim_size(1) == numParamsBasis, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects the rigth number of "
                    "parameters each for each basis function."));
                OP_REQUIRES(pContext, inGradients.dims() == 3 && 
                    inGradients.shape().dim_size(0) == numSamples &&
                    inGradients.shape().dim_size(1) == numInFeatures &&
                    inGradients.shape().dim_size(2) == numBasis, 
                    errors::InvalidArgument("BasisProjBilGradsOp expects the rigth number of feaure gradients,"));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* output1GPUPtr = nullptr;
                float* output2GPUPtr = nullptr;
                TensorShape outShape1 = TensorShape{numPts, numInFeatures};
                TensorShape outShape2 = TensorShape{numBasis, numParamsBasis};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape1, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (1, pContext, outShape2, &output2GPUPtr));

                //Compute the convolution gradients.
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, numNeighVals, 
                    mccnn::basis_proj_grads_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, numPts, numSamples, numNeighbors, 
                    numInFeatures, inPtsGPUPtr, inPtFeaturesGPUPtr, inSamplesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inInvRadiiGPUPtr, inBasisGPUPtr, 
                    inPDFsGPUPtr, inNeighValsGPUPtr, inGradientsGPUPtr, output1GPUPtr, output2GPUPtr,
                    nullptr, nullptr, nullptr, nullptr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };

    /**
     *  Operation to compute a monte carlo convolution.
     */
    class BasisProjBilGradsWithPtGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjBilGradsWithPtGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("basis_type", &basisType_));
                OP_REQUIRES(pContext, basisType_ >= 6 && basisType_ < 10, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp requires a valid basis type."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inSamples = pContext->input(2); 
                const Tensor& inNeighbors = pContext->input(3); 
                const Tensor& inSampleNeighIndices = pContext->input(4);
                const Tensor& inInvRadii = pContext->input(5);
                const Tensor& inPDFs = pContext->input(6);
                const Tensor& inNeighVals = pContext->input(7);
                const Tensor& inBasis = pContext->input(8);
                const Tensor& inGradients = pContext->input(9);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);
                unsigned int numNeighVals = inNeighVals.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inPtsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const float* inSamplesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSamples);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inNeighValsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inNeighVals);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);
                const float* inGradientsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGradients);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions, numNeighVals);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inSamples.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a number of dimensions in"
                    " inSamples equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inPtFeatures.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a number of points in"
                    " inPtFeatures equal to the number of points in the input points tensor"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inSampleNeighIndices.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects the same number of points "
                    "in inSampleNeighIndices as in the samples tensor."));
                OP_REQUIRES(pContext, inPDFs.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a number of pdf values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inNeighVals.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects a number of neighbor values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inBasis.dims() == 2 && 
                    inBasis.shape().dim_size(1) == numParamsBasis, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects the rigth number of "
                    "parameters each for each basis function."));
                OP_REQUIRES(pContext, inGradients.dims() == 3 && 
                    inGradients.shape().dim_size(0) == numSamples &&
                    inGradients.shape().dim_size(1) == numInFeatures &&
                    inGradients.shape().dim_size(2) == numBasis, 
                    errors::InvalidArgument("BasisProjBilGradsWithPtGradsOp expects the rigth number of feaure gradients,"));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* output1GPUPtr = nullptr;
                float* output2GPUPtr = nullptr;
                float* output3GPUPtr = nullptr;
                float* output4GPUPtr = nullptr;
                float* output5GPUPtr = nullptr;
                float* output6GPUPtr = nullptr;
                TensorShape outShape1 = TensorShape{numPts, numInFeatures};
                TensorShape outShape2 = TensorShape{numBasis, numParamsBasis};
                TensorShape outShape3 = TensorShape{numPts, numDimensions};
                TensorShape outShape4 = TensorShape{numSamples, numDimensions};
                TensorShape outShape5 = TensorShape{numNeighbors};
                TensorShape outShape6 = TensorShape{numNeighbors, numNeighVals};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape1, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (1, pContext, outShape2, &output2GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (2, pContext, outShape3, &output3GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (3, pContext, outShape4, &output4GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (4, pContext, outShape5, &output5GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (5, pContext, outShape6, &output6GPUPtr));

                //Compute the convolution gradients.
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, numNeighVals, 
                    mccnn::basis_proj_grads_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, numPts, numSamples, numNeighbors, 
                    numInFeatures, inPtsGPUPtr, inPtFeaturesGPUPtr, inSamplesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inInvRadiiGPUPtr, inBasisGPUPtr, 
                    inPDFsGPUPtr, inNeighValsGPUPtr, inGradientsGPUPtr, output1GPUPtr, output2GPUPtr,
                    output3GPUPtr, output4GPUPtr, output5GPUPtr, output6GPUPtr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };
}

REGISTER_KERNEL_BUILDER(Name("BasisProjBil").Device(DEVICE_GPU), mccnn::BasisProjBilOp);
REGISTER_KERNEL_BUILDER(Name("BasisProjBilGrads").Device(DEVICE_GPU), mccnn::BasisProjBilGradsOp);
REGISTER_KERNEL_BUILDER(Name("BasisProjBilGradsWithPtGrads").Device(DEVICE_GPU), mccnn::BasisProjBilGradsWithPtGradsOp);