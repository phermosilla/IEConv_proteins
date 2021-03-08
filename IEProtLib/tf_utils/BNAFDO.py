'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file BNAFDO.py

    \brief Object to create bach normalization, activation function, and 
        dropout layers.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf

class BN_AF_DO:
    """Layer that applies Batch Normalization, Activation Function, and Drop
        Out to an input tensor.

    Attributes:
        isTraining_ (bool tensor): Indicates if we are execution training or
            testing graph.
        BNMomentum_ (float tensor): Batch normalization momentum.
        activationFunction_ (activation function): Activation function used.
        doRate_ (float tensor): Probability to turn of a feature.
        rndNoise_ (float): Noise added to the features.
    """
    
    def __init__(self,
        pIsTraining, 
        pNormType,
        pBNMomentum,
        pBRNMomentum,
        pBRNClipping,
        pActivationFunction = tf.nn.elu,
        pDORate = 0.5,
        noiseLevel = 0.0):
        """Constructor.

        Args:
            pIsTraining (bool tensor): Indicates if we are execution training or
                testing graph.
            pInFeatures (float tensor nxf): Input features.
            pNormType (string): Type of normalization used.
            pBNMomentum (float tensor): Batch normalization momentum.
            pBRNMomentum (float tensor): Batch renormalization momentum.
            pBRNClipping (dictionary): Dictionary with the batch renormalization parameters.
            pActivationFunction (activation function): Activation function used.
            pDORate (float): Probability to turn of a feature.
            noiseLevel (float): Noise level added to the features.
        """
        self.isTraining_ = pIsTraining
        self.normType_ = pNormType
        self.BNMomentum_ = pBNMomentum
        self.BRNMomentum_ = pBRNMomentum
        self.BRNClipping_ = pBRNClipping
        self.activationFunction_ = pActivationFunction
        self.doRate_ = pDORate
        self.rndNoise_ = noiseLevel        

    
    def __call__(self, pInFeatures, pLayerName, 
        pApplyBN = True, pApplyNoise = True, 
        pApplyAF = True, pApplyDO = True):
        """Method to apply Batch ReNormalization / Activation Function / Drop Out 
            to a set of features.

        Args:
            pInFeatures (float tensor nxf): Input features.
            pLayerName (string): Layer name.
            pApplyBN (bool): Boolean that indicates if the batch norm is applied.
            pApplyNoise (bool): Boolean that indicates if the noise is applied.
            pApplyAF (bool): Boolean that indicates if the activation function is applied.
            pApplyDO (bool): Boolean that indicates if the dropout is applied.
        Returns:
            (float tensor nxf): Result of applying these operations to the input 
                features.
        """

        #Create the layer name.
        outLayerName = pLayerName
        if pLayerName is None:
            outLayerName = pInFeatures.name +"_BN_AF_DO"

        inFeatures = pInFeatures

        #Apply the batch renormalization.
        if pApplyBN:
            if self.normType_ == "batchnorm":
                inFeatures = tf.layers.batch_normalization(
                    inputs = inFeatures, 
                    momentum = self.BNMomentum_, 
                    trainable = True, 
                    training = self.isTraining_, 
                    name = outLayerName+"_BN")
            elif self.normType_ == "batchrenorm":
                inFeatures = tf.layers.batch_normalization(
                    inputs = inFeatures, 
                    momentum = self.BNMomentum_, 
                    trainable = True, 
                    training = self.isTraining_, 
                    name = outLayerName+"_BRN",
                    renorm=True,
                    renorm_clipping = self.BRNClipping_,
                    renorm_momentum = self.BRNMomentum_)
            elif self.normType_ == "l2norm":
                inFeatures = tf.math.l2_normalize(inFeatures, axis=-1)

        #Add noise.
        if pApplyNoise:
            if self.rndNoise_ > 0.0:
                noise = tf.cond(self.isTraining_, 
                    true_fn = lambda: tf.random.normal(tf.shape(inFeatures), 0.0, self.rndNoise_),
                    false_fn = lambda: tf.zeros(tf.shape(inFeatures), dtype=tf.float32))
                inFeatures = inFeatures + noise

        #Apply the activation function.
        if pApplyAF:
            inFeatures = self.activationFunction_(inFeatures, name = outLayerName+"_AF")

        #Apply dropout.
        if pApplyDO:
            if self.doRate_ > 0.0:
                tfDORate = tf.cond(self.isTraining_, 
                    true_fn = lambda: self.doRate_,
                    false_fn = lambda: 0.0)
                inFeatures = tf.nn.dropout(inFeatures, 1.0 - tfDORate, 
                    name=outLayerName+"_DO")

        return inFeatures


def BNAFDO_from_config_file(
    pPrefixStr, 
    pConfig, 
    pIsTraining, 
    pEpochStep,
    pMaxEpoch):
    """Constructor.

        Args:
            pPrefixStr (stirng): Prefix string in the parameters.
            pConfig (dictionary): Dictionary with the configuration parameters.
            pIsTraining (bool tensor): Tensor that indicates if we are in 
                training or evaluation mode.
            pEpochStep (int tensor): Tensor with the current epoch counter.
            pMaxEpoch (int): Maximum number of epochs.
        Return:
            (BN_AF_DO): BN_AF_DO object.
        """

    #Get the BRN_AF_DO parameters.
    dropoutRate = float(pConfig[pPrefixStr+'.dropoutrate'])
    randomNoise = float(pConfig[pPrefixStr+'.noiselevel'])
    normType = pConfig[pPrefixStr+'.normtype']
    if normType == "batchnorm" or normType == "batchrenorm":
        bnInit = float(pConfig[pPrefixStr+'.bninit'])
        bnDecayRate = int(pConfig[pPrefixStr+'.bndecayrate'])
        bnDecayFactor = float(pConfig[pPrefixStr+'.bndecayfactor'])
        bnMin = float(pConfig[pPrefixStr+'.bnmin'])
        if normType == "batchrenorm":
            numEpochsBN = int(pConfig[pPrefixStr+'.numepochsbn'])
            brnRMax = float(pConfig[pPrefixStr+'.brnrmax'])
            brnDMax = float(pConfig[pPrefixStr+'.brndmax'])
    else:
        bnInit = 0.0
        bnDecayRate = 0
        bnDecayFactor = 0
        bnMin = 0.0
    activationFunct = None
    if pConfig[pPrefixStr+'.activation'] == 'RELU':
        activationFunct = tf.nn.relu
    elif pConfig[pPrefixStr+'.activation'] == 'LRELU':
        activationFunct = tf.nn.leaky_relu
    elif pConfig[pPrefixStr+'.activation'] == 'ELU':
        activationFunct = tf.nn.elu


    #Create the tensorflow variables.
    droprateTF = tf.cast(pIsTraining, dtype=tf.float32)*dropoutRate

    bnDecayExp = tf.train.exponential_decay(bnInit, pEpochStep, 
        bnDecayRate, bnDecayFactor, staircase=True)
    bnDecayExp = 1.0 - tf.maximum(bnDecayExp, bnMin)

    if normType == "batchrenorm":
        endEpoch = pMaxEpoch-numEpochsBN
        lerpBRN = 1.0 - tf.clip_by_value((tf.cast(pEpochStep, tf.float32)
            -float(numEpochsBN))/float(endEpoch), 0.0, 1.0)
        rmax = lerpBRN + (1.0-lerpBRN)*brnRMax
        dmax = (1.0-lerpBRN)*brnDMax
        brnClipping = { 'rmax': rmax,
                        'rmin': 1.0/rmax,
                        'dmax': dmax}
    else:
        brnClipping = {}

    #Create the final object.
    return BN_AF_DO(
        pIsTraining = pIsTraining,
        pNormType = normType,
        pBNMomentum = bnDecayExp,
        pBRNMomentum = bnDecayExp,
        pBRNClipping = brnClipping,
        pActivationFunction = activationFunct,
        pDORate = dropoutRate,
        noiseLevel = randomNoise)