'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ProtEncoder.py

    \brief Object to encode a protein.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import tensorflow as tf

from IEProtLib.mol import Molecule
from IEProtLib.mol import Protein
from IEProtLib.mol import MolConvBuilder
from IEProtLib.tf_utils import PoolFeatureMode
from IEProtLib.tf_utils import BN_AF_DO, BNAFDO_from_config_file
from IEProtLib.py_utils import parse_elem_list

class ProtEncoder:
    """Encoder.

    Attributes:
        config_ (dictionary): Dictionary with the configuration parameters.
        convBuilder_ (MolConvBuilder): Convolution builder.
        protein_ (Protein): Protein.
        outLayers_ (list of tensors): Output of each level of the encoder.
        latentCode_ (float tensor bxf): Latent code for each model of the batch.
        numConvs_ (int): Number of convolutions.
        numMCGraph_ (int): Number of graph convolutions.
        numMCConvs_ (int): Number of monte carlo convolutions.
    """

    def __init__(self, pConfig, pConvBuilder, pProtein, 
        pInFeatures, pBNAFDO):
        """Constructor.

        Args:
            pConfig (dictionary): Dictionary containing the parameters 
                needed to create the network in string format.
            pConvBuilder (MolConvBuilder): Convolution builder.
            pProtein (Protein): Protein.
            pInFeatures (float tensor nxf): Input features.
            pBNAFDO (BNAFDO): BNAFDO object.
        """
        self.config_ = pConfig
        self.convBuilder_ = pConvBuilder
        self.protein_ = pProtein


        print("")
        print("##################### ProtEncoder #####################")
        print("")

        ######### GET THE CONFIGURATION PARAMETERS
        convRadii = parse_elem_list(self.config_['enc.radii'], float)
        numBasis = int(self.config_['enc.numbasis'])
        featureList = parse_elem_list(self.config_['enc.numfeatures'], int)
        numBlocksList = parse_elem_list(self.config_['enc.numblocks'], int)
        globalPoolingStr = self.config_['enc.globalpool']
        if 'enc.atomdo' in self.config_:
            atomDo = float(self.config_['enc.atomdo'])
        else:
            atomDo = 0.0
        modelType = self.config_['enc.modeltype']
        if 'enc.initconv' in self.config_:
            initConv = self.config_['enc.initconv'] == "True"
        else:
            initConv = True
        if globalPoolingStr == "max":
            globalPooling = PoolFeatureMode.maximum
        elif globalPoolingStr == "avg":
            globalPooling = PoolFeatureMode.average
        elif globalPoolingStr == "att":
            globalPooling = PoolFeatureMode.attention
    

        ######### Define the output of each layer for possible skip links in a decoder.
        self.outLayers_ = []

        ######### CREATE THE NETWORK
        if modelType == "resnetb":
            self.numConvs_, self.num1x1Convs_, self.numMolConvs_ = \
                self.__create_molconv_resnet_model__(
                pInFeatures, numBasis, pBNAFDO, convRadii, 
                numBlocksList, featureList, globalPooling, 
                initConv, atomDo)


    def __create_molconv_resnet_model__(self,
        pInFeatures, pNumBasis, pBNAFDO, 
        pConvRadii, pNumBlocksList, pFeatureList, 
        pGlobalPooling, pInitConv, pAtomDropOut):
        """Method to create a molconv resnet model.

        Args:
            pInFeatures (float tensor nxf): Input features.
            pNumBasis (int): Number of basis.
            pBNAFDO (BNAFDO): BNAFDO object.
            pConvRadii (float list): Convolution radius.
            pNumBlocksList (int list): List of number of blocks.
            pFeatureList (int list): Number of features for each level.
            pGlobalPooling (PoolFeatureMode): Global pooling mode.
            pInitConv (bool): Boolean that indicates if we will use an initial
                convolution.
            pAtomDropOut (float): Probability of removing an atom during training.
        """

        print("############ ResnetB enconder")
        print()

        # Compute the number of levels.
        numLevels = len(pConvRadii)

        accumNumConvs = 0
        accumNum1x1Convs = 0
        accumNumMolConvs = 0

        # First convolution.
        if pInitConv:
            curFeatures = self.convBuilder_.create_1x1_convolution(pInFeatures, 
                pFeatureList[0], "Init_convolution")
            accumNumConvs = 1
            accumNum1x1Convs = 1
        else:
            curFeatures = pInFeatures

        # Create the network.
        initLayer = 0
        for curLevel in range(numLevels):

            print("############ Level", curLevel)
            print("Num blocks:", pNumBlocksList[curLevel])
            print("Num features:", pFeatureList[curLevel])
            print("Radius:", pConvRadii[curLevel])

            # Create the blocks.
            curFeatures = self.convBuilder_.create_molconv_resnet_blocks( 
                pMolecule = self.protein_.molObjects_[curLevel],
                pInFeatures = curFeatures, 
                pNumBlocks = pNumBlocksList[curLevel], 
                pOutNumFeatures = pFeatureList[curLevel], 
                pRadii = pConvRadii[curLevel], 
                pBNAFDO = pBNAFDO,
                pConvName = "Block_Level_"+str(curLevel),
                pNumBasis = pNumBasis,
                pAtomDropOut = pAtomDropOut)

            print("")

            # Store the output of the layers.
            self.outLayers_.append(curFeatures)

            # Increase the counter of convolutions.
            accumNumConvs += pNumBlocksList[curLevel]*3 + 1
            accumNum1x1Convs += pNumBlocksList[curLevel]*2 + 1
            accumNumMolConvs += pNumBlocksList[curLevel]

            # Global pooling.
            if curLevel == numLevels-1:
                curFeatures = self.convBuilder_.create_global_feature_pooling(
                    curFeatures, self.protein_.molObjects_[curLevel].batchIds_,
                    self.protein_.molObjects_[curLevel].batchSize_,
                    pBNAFDO, pGlobalPooling)
            # Normal pooling between molecular representations.
            else:
                curFeatures = self.convBuilder_.create_prot_pooling(
                    curFeatures, self.protein_, curLevel+1, pBNAFDO)                               
            
        #Save the global latent vector.
        self.latentCode_ = curFeatures

        #Return the number of convolutions used.
        return accumNumConvs, accumNum1x1Convs, accumNumMolConvs
        
