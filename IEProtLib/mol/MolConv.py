'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MolConv.py

    \brief Convolution operation for a protein.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import math
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from IEProtLibModule import basis_proj_bilateral

class MolConv:
    """Factory to create a MolConv.
    
    Attributes:
        staticRandomState_ (numpy.random.RandomState): Static random state.
    """

    staticRandomState_ = np.random.RandomState(None)


    def create_convolution(self, 
        pConvName,
        pNeighborhood, 
        pTopoDists,
        pFeatures,
        pNumOutFeatures,
        pWeightRegCollection,
        pNumBasis,
        pTopoOnly = False,
        pUse3DDistOnly = False):
        """Function to create a spatial convolution.

        Args:
            pConvName (string): String with the name of the convolution.
            pNeighborhood (MCNeighborhood): Input neighborhood.
            pTopoDists (float tensor nxu): Topological distances of each neighbor.
            pFeatures (float tensor nxf): Input point cloud features.
            pNumOutFeatures (int): Number of output features.
            pWeightRegCollection (string): Weight regularization collection name.
            pNumBasis (int): Number of basis functions.
            pTopoOnly (bool): Boolean that indicates if only the geodesic distances are used.
            pUse3DDistOnly (bool): Boolean that indicates if the 3D distance is used instead
                of the axis difference.

        Returns:
            n'xpNumOutFeatures tensor: Tensor with the result of the convolution.
        """

        #Check if the number of projection vectors has an acceptable value.
        if pNumBasis != 8 and pNumBasis != 16 and pNumBasis != 32:
            raise RuntimeError('The number of basis vectors should be 8, 16, or 32.')

        #Compute the number of kernels.
        numInFeatures = pFeatures.shape.as_list()[1]
        numKernels = numInFeatures*pNumOutFeatures

        #Compute the weight initialization parameters.
        numDims = pNeighborhood.pcSamples_.pts_.shape.as_list()[1]
        numNeighVals = pTopoDists.shape.as_list()[1]

        #If we use the distance in 3D.
        if pUse3DDistOnly:
            neighPts = tf.gather(pNeighborhood.grid_.sortedPts_, pNeighborhood.neighbors_[:,0])
            centerPts = tf.gather(pNeighborhood.pcSamples_.pts_, pNeighborhood.neighbors_[:,1])
            diffPts = (neighPts - centerPts)/pNeighborhood.radii_
            lengthPts = tf.norm(diffPts, axis = -1, keepdims = True)
            pTopoDists = tf.concat([pTopoDists, lengthPts], axis = -1)
            numNeighVals = numNeighVals + 1 
            pTopoOnly = True

        #Define the tensorflow variable.
        if pTopoOnly:
            stdDev = math.sqrt(1.0/float(numNeighVals))
            hProjVecTF = tf.get_variable(pConvName+'_half_proj_vectors', shape=[pNumBasis, numNeighVals], 
                initializer=tf.initializers.truncated_normal(stddev=stdDev), dtype=tf.float32, trainable=True)
            hProjVecTF = tf.concat([tf.zeros([pNumBasis,numDims], dtype=tf.float32), hProjVecTF], axis = -1)
        else:
            stdDev = math.sqrt(1.0/float(numDims+numNeighVals))
            hProjVecTF = tf.get_variable(pConvName+'_half_proj_vectors', shape=[pNumBasis, numDims+numNeighVals], 
                initializer=tf.initializers.truncated_normal(stddev=stdDev), dtype=tf.float32, trainable=True)

        hProjBiasTF = tf.get_variable(pConvName+'_half_proj_biases', shape=[pNumBasis, 1], 
            initializer=tf.initializers.zeros(), dtype=tf.float32, trainable=True)
        basisTF = tf.concat([hProjVecTF, hProjBiasTF], axis = 1)

        #Create the weights.
        stdDev = math.sqrt(2.0/float(pNumBasis*numInFeatures))
        weights = tf.get_variable(pConvName+'_conv_weights', shape=[pNumBasis * numInFeatures, pNumOutFeatures], 
            initializer=tf.initializers.truncated_normal(stddev=stdDev), dtype=tf.float32, trainable=True)
            
        #Get the input features projected on the kernel point basis.
        inWeightFeat = basis_proj_bilateral(pNeighborhood, pTopoDists, 
            pFeatures, basisTF, 7, True)
            
        #Compute the convolution.
        convFeatures = tf.matmul(tf.reshape(inWeightFeat, [-1, numInFeatures*pNumBasis]), weights)

        #Add to collection for weight regularization.
        tf.add_to_collection(pWeightRegCollection, tf.reshape(weights, [-1]))

        return convFeatures
