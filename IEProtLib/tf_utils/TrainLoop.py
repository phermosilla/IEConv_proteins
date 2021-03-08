'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file TrainLopp.py

    \brief Train loop.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import time
import numpy as np
import configparser
import abc 
from six import with_metaclass

import tensorflow as tf

#DEBUG
#from tensorflow.python import debug as tf_debug


current_milli_time = lambda: time.time() * 1000.0

class TrainLoop(with_metaclass(abc.ABCMeta)):
    """Class to train over several epochs.

    Attributes:
        logFolder_ (string): Path to the log folder.
        numEpochs_ (int): Number of epochs.
        numEpochsEval_ (int): Number of epochs between each evaluation.
        gpuId_ (string): Id of the GPU Device used.
        gpuMem_ (float): Percentage of the gpu memory used.
        epochStep_ (int tensor): Tensor with the current epoch id.
        epochStepIncr_ (tf operation): Tensorflow operation to increment the epoch 
            step.
        sess_ (Session): Tensorflow session.
        summaryWriter_ (FileWriter): Writer used to save the summaries.
    """
    
    def __init__(self, pConfig):
        """Constructor.

        Args:
            pConfig (dictionary): Dictionary with the configuration of the 
                train loop.
        """

        #Load the train loop parameters.
        self.logFolder_ = pConfig['logfolder']
        self.numEpochs_ = int(pConfig['numepochs'])
        self.numEpochsEval_ = int(pConfig['numepochseval'])
        self.gpuId_ = pConfig['gpuid']
        self.gpuMem_ = float(pConfig['gpumem'])
        if 'numthreads' in pConfig:
            self.numThreads_ = int(pConfig['numthreads'])
        else:
            self.numThreads_ = None

        #Create the log folder.
        if not os.path.exists(self.logFolder_): os.mkdir(self.logFolder_)

        #Create the epoch step counter.
        self.epochStep_ = tf.Variable(0, name='epoch_step', trainable=False)
        self.epochStepIncr_ = tf.assign(self.epochStep_, self.epochStep_+1)

        #Initialize the train loop.
        self.__create_datasets__()
        self.__create_model__()
        self.__create_tf_session__()
        self.__create_trainers__()
        self.__create_savers__()
        self.__create_tf_summaries__()


    @abc.abstractmethod
    def __create_datasets__(self):
        """Method to create the datasets.
        """
        pass


    @abc.abstractmethod
    def __create_model__(self):
        """Method to create the model.
        """
        pass


    def __create_tf_session__(self):
        """Method to create the tensorflow session and other tensorflow objects.
        """

        #Create session
        gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuMem_, 
            visible_device_list=self.gpuId_)
        tfConfig = tf.ConfigProto(gpu_options=gpuOptions)
        if not(self.numThreads_ is None):
            tfConfig.intra_op_parallelism_threads = self.numThreads_
            tfConfig.inter_op_parallelism_threads = self.numThreads_
        self.sess_ = tf.Session(config=tfConfig)

        #DEBUG
        #self.sess_ = tf_debug.LocalCLIDebugWrapperSession(self.sess_)
        #self.sess_.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #Create the summary writer
        self.summaryWriter_ = tf.summary.FileWriter(self.logFolder_, self.sess_.graph)
        self.summaryWriter_.add_graph(self.sess_.graph)
            

    @abc.abstractmethod
    def __create_trainers__(self):
        """Method to create the trainer objects.
        """
        pass

    
    @abc.abstractmethod
    def __create_savers__(self):
        """Method to create the saver objects.
        """
        pass
        

    @abc.abstractmethod
    def __create_tf_summaries__(self):
        """Method to create the tensorflow summaries.
        """
        pass


    @abc.abstractmethod
    def __train_one_epoch__(self, pNumEpoch):
        """Private method to train one epoch.

        Args:
            pNumEpoch (int): Current number of epoch.
        """
        pass

    
    @abc.abstractmethod
    def __test_one_epoch__(self, pNumEpoch):
        """Private method to test one epoch.

        Args:
            pNumEpoch (int): Current number of epoch.
        """
        pass


    def train(self):
        """Method to train the model.
        """

        #Init variables.
        self.sess_.run(tf.global_variables_initializer())
        self.sess_.run(tf.local_variables_initializer())
        
        #Iterate over the epochs.
        for epochIter in range(self.numEpochs_+1):

            print("")
            print("Epoch", epochIter, "/", self.numEpochs_)

            #Get the starting time of the epoch.
            startEpoch = current_milli_time()

            #Train one epoch.
            self.__train_one_epoch__(epochIter)

            #Get the end time of the epoch.
            endTrainEpoch = current_milli_time()

            print("End train %.6f sec" % ((endTrainEpoch-startEpoch)/1000.0))

            #Only test every certain number of epochs.
            if epochIter%self.numEpochsEval_ == 0:
                self.__test_one_epoch__(epochIter)            

            #Increment epoch counter.
            self.sess_.run(self.epochStepIncr_)

            #Get the end time of the epoch.
            endEpoch = current_milli_time()

            print("End epoch %.6f sec" % ((endEpoch-startEpoch)/1000.0))
            print("")