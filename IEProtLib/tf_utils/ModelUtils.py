'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ModuleUtils.py

    \brief Util functions.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_num_parameters(pScope=None):
    """Method to get the number of parameters of a model.

    Args:
        pScope (string): Scope.
    Returns:
        (int): Number of parameters.
    """
    totalParams = 0
    if not(pScope is None):
        trainVars = tf.trainable_variables()
    else:
        trainVars = tf.trainable_variables(pScope)
    for variable in trainVars:  
        localParams=1
        shape = variable.get_shape()
        for i in shape:
            localParams*=i.value    
        totalParams+=localParams
    return totalParams
    

def tensors_in_checkpoint_file(pFileName):
    """Function to get the tensor name in a checkpoint file.

    Args:
        pFileName (string): Checkpoint file path.
    Returns:
        string list: Names of tensors in a checkpoint.
    """
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(pFileName)
    varToShapeMap = reader.get_variable_to_shape_map()
    for key in sorted(varToShapeMap):
        varlist.append(key)
    return varlist


def get_tensors_list(pTensorList):
    """Function to get a set of tensor variables.

    Args:
        pTensorList (string list): List of tensor names.
    Returns:
        tensor list: Tensor variables.
    """
    
    varList = dict()
    for i, tensorName in enumerate(pTensorList):
        try:
            tensorAux = tf.get_default_graph().get_tensor_by_name(tensorName+":0")
            varList[tensorName] = tensorAux
        except:
            pass
    return varList