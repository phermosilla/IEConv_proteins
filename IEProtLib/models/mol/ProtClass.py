'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ProtClass.py

    \brief Classificaiton model of proteins.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import tensorflow as tf

from IEProtLib.mol import ProteinPH
from IEProtLib.mol import MolConvBuilder
from IEProtLib.tf_utils import BN_AF_DO, BNAFDO_from_config_file
from IEProtLib.tf_utils import get_num_parameters, cross_entropy, binary_cross_entropy
from IEProtLib.py_utils import parse_elem_list
from IEProtLib.py_utils.py_mol import PyPeriodicTable
from IEProtLib.models.mol import ProtEncoder

class ProtClass:
    """Classification network ProtClass.

    Attributes:
        config_ (dictionary): Dictionary with the configuration parameters.
        batchSize_ ( int): Batch size.
        numDims_ (int): Number of dimensions.
        numOutClasses_ (int): Number of output classes.
        numInputFeatures_ (int): Number of features the user has to provide.
        
        ptsPH_ (placeholder float nxd): Place holder with the input points.
        batchIdsPH_ (placeholder int n): Place holder with the input batch ids.
        
        featuresPH_ (placeholder float nxnumNeurons): Place holder with the input 
            features.
        labelsPH_ (placeholder int n): Place holder with the input labels.
        isTrainingPH_ (placeholder bool): Place holder with the boolean value
            that determines if we are training or evaluating our model.
        encoder_ (MCProtEncoder): Encoder.
        logits_ (float tensor bxo): Logits of each model of the batch.        
        convBuilder_ (MolConvBuilder): Convolution builder.
        xEntropyLoss_ (float tensor): Cross entropy loss.
        kIntRegLoss_ (float tensor): Kernel integral regularization loss.
    """

    def __init__(self, 
        pConfig, 
        pNumDims, 
        pBatchSize, 
        pNumOutClasses, 
        pAminoInput = False):
        """Constructor.

        Args:
            pConfig (dictionary): Dictionary containing the parameters 
                needed to create the network in string format.
            pNumDims (int): Number of dimensions.
            pBatchSize (int): Size of each batch.
            pNumOutClasses (int): Number of output classes.
            pAminoInput (bool): Boolean that indicates if we use as input
                the aminoacid layer.
        """
        # Save parameters
        self.config_ = pConfig
        self.numDims_ = pNumDims
        self.batchSize_ = pBatchSize
        self.numOutClasses_ = pNumOutClasses
        self.aminoInput_ = pAminoInput

        # Get parameters from config dictionary.
        self.numFeaturesLastLayer_ = int(self.config_['numfeatlastlayer'])

    
    def create_placeholders(self, pNumInitFeatures):
        """Method to create the placeholders.

        Args:
            pNumInitFeatures (int): Number of input features the user
                wants to provide.
        Return:
            (int): Number of input features required by the model.
        """
        #Calculate the number of input features.
        self.numInputFeatures_ = pNumInitFeatures

        #Create the placeholders.
        self.proteinPH_ = ProteinPH(self.numDims_, self.batchSize_, self.aminoInput_, self.config_)
        if pNumInitFeatures > 0:
            self.featuresPH_ = tf.placeholder(tf.float32, [None, self.numInputFeatures_])
        self.labelsPH_ = tf.placeholder(tf.float32, [None, self.numOutClasses_])
        self.isTrainingPH_ = tf.placeholder(tf.bool, shape=())

        #Return the number of features.
        return self.numInputFeatures_


    def associate_inputs_to_ph(self, 
        pDict, pProteinBatch, pFeatures, pLabels, pIsTraining):
        """Method to associate inputs to placeholders in the dictionary pDict.

        Args:
            pDict (dictionary): Dictionary to fill.
            pProteinBatch (PyProteinBatch): Protein batch.
            pFeatures (numpy array float nxf): Input point features.
            pLabels (numpy array int n): Input labels.
            pIsTraining (bool): Boolean that indicates if we are training or not.
        """
        self.proteinPH_.update_dictionary(pDict, pProteinBatch)
        if self.numInputFeatures_ > 0:
            pDict[self.featuresPH_] = pFeatures
        pDict[self.labelsPH_] = pLabels
        pDict[self.isTrainingPH_] = pIsTraining

    
    def create_model(self, pEpochStep, pMaxEpoch):
        """Method to create the model.

        Args:
            pEpochStep (int tensor): Tensor with the current epoch counter.
            pMaxEpoch (int): Maximum number of epochs.
        """

        print("")
        print("")
        print("")
        print("##################### MCProtMultiClass #####################")
        print("")

        ######### CREATE THE CONVOLUTION BUILDER OBJECT
        self.convBuilder_ = MolConvBuilder("weight_regularization_collection")

        ######### CREATE THE MCBRNAFDO OBJECTS
        BNAFDOConv = BNAFDO_from_config_file('convbnafdo', self.config_, 
            self.isTrainingPH_, pEpochStep, pMaxEpoch)
        BNAFDOFull = BNAFDO_from_config_file('fullbnafdo', self.config_, 
            self.isTrainingPH_, pEpochStep, pMaxEpoch)

        ######### PREPARE THE INPUT FEATURES
        auxPT = PyPeriodicTable()
        if self.aminoInput_:
            self.embeddingAtomTypes_ = tf.get_variable("EmbeddingAminoTypes", 
                [auxPT.get_num_aminoacids(), max(self.numInputFeatures_, 3)], 
                initializer=tf.random_uniform_initializer())
        else:
            self.embeddingAtomTypes_ = tf.get_variable("EmbeddingAtomTypes", 
                [auxPT.get_num_atoms(), max(self.numInputFeatures_, 3)], 
                initializer=tf.random_uniform_initializer())
        inFeatures = tf.nn.embedding_lookup(self.embeddingAtomTypes_, self.proteinPH_.atomTypesPH_)
        if self.numInputFeatures_ > 0:
            inFeatures = tf.concat([self.featuresPH_, inFeatures], axis=-1)
            
        ######### CREATE THE NETWORK
        self.encoder_ = ProtEncoder(self.config_, self.convBuilder_, 
            self.proteinPH_, inFeatures, BNAFDOConv)

        #Last fully connected layers.
        if self.numFeaturesLastLayer_ > 0:
            fc1 = BNAFDOFull(self.encoder_.latentCode_, "Full_1_BAD")
            fc1 = self.convBuilder_.create_1x1_convolution(fc1, self.numFeaturesLastLayer_, "Full_1")
            fc2 = BNAFDOFull(fc1, "Full_2_BAD")
        else:
            fc1 = self.encoder_.latentCode_
            fc2 = BNAFDOFull(fc1, "Full_2_BAD", 
                    pApplyBN = True, pApplyNoise = False, pApplyAF = True, pApplyDO = False)
        self.logits_ = self.convBuilder_.create_1x1_convolution(fc2, 
            self.numOutClasses_, "Full_2")

        self.predictions_ = tf.sigmoid(self.logits_)

        #Get the number of trainable parameters
        totalParams = get_num_parameters()

        #Print final statistics
        print("############ Number of convolutions:", self.encoder_.numConvs_+3)
        print("############ Number of 1x1 convolutions:", self.encoder_.num1x1Convs_+3)
        print("############ Number of mol convolutions:", self.encoder_.numMolConvs_)
        print("############ Number of parameters:", totalParams)
        print("")
        print("")
        print("")


    def create_loss(self, pWeights = None, pCluterMatricesLoss = 0.0):
        """Method to get the loss.

        Return:
            tensor float: Cross entropy loss.
        """

        #Compute the cross entropy loss.
        auxLabelsIndex = tf.reshape(tf.argmax(self.labelsPH_, axis= -1), [-1])
        if pWeights is None:
            auxWeights = None
        else:
            auxWeights = tf.gather(pWeights, tf.reshape(auxLabelsIndex, [-1]))
        if self.numOutClasses_ > 1:
            self.xEntropyLoss_ = cross_entropy(auxLabelsIndex, self.logits_, pWeights = auxWeights)
        else:
            self.xEntropyLoss_ = binary_cross_entropy(self.labelsPH_, self.logits_, pPosWeight = auxWeights)

        return self.xEntropyLoss_
