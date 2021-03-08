'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Test.py

    \brief Code to test a classification network on the function task.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import logging
import math
import time
import numpy as np
import configparser
import argparse

import tensorflow as tf
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.dirname(BASE_DIR)
ROOT_PROJ_DIR = os.path.dirname(TASKS_DIR)
sys.path.append(ROOT_PROJ_DIR)

from IEProtLib.tf_utils import TestLoop, tensors_in_checkpoint_file
from IEProtLib.models.mol import ProtClass
from IEProtLib.py_utils import visualize_progress
from Datasets import ProtFunctDataSet

current_milli_time = lambda: time.time() * 1000.0

class ProtFunctTestLoop(TestLoop):
    """Class to test a function prediction network.
    """
    
    def __init__(self, pConfigFile):
        """Constructor.

        Args:
            pConfigFile (string): Path to the configuration file.
        """

        #Load the configuration file.
        self.config_ = configparser.ConfigParser()
        self.config_.read(pConfigFile)

        #Load the parameters.
        trainConfigDict = self.config_._sections['ProtFunct']
        self.batchSize_ = int(trainConfigDict['batchsize'])
        self.augment_ = trainConfigDict['augment'] == "True"
        self.checkPointName_ = trainConfigDict['checkpointname']
        self.logFolder_ = trainConfigDict['logfolder']
        self.aminoInput_ = trainConfigDict['aminoinput'] == "True"
        self.topK_ = int(trainConfigDict['topk'])

        #Call the constructor of the parent.
        TestLoop.__init__(self, self.config_._sections['TestLoop'])


    def __create_datasets__(self):
        """Method to create the datasets.
        """

        print("")
        print("########## Loading test dataset")
        self.testDS_ = ProtFunctDataSet("Test",
            "../../Datasets/data/ProtFunct/", 
            pAminoInput = self.aminoInput_,
            pRandSeed = 33,
            pPermute=False,
            pLoadText = False)
        print(self.testDS_.get_num_proteins(), "proteins loaded")

        # Create the accumulative logits.
        self.accumLogits_ = np.full((self.testDS_.get_num_proteins(), self.__get_num_terms__()), 0.0, dtype=np.float64)

        #Declare the array to store the accuracy of each class each iteration.
        self.votesXClassAcc_ = np.full((self.__get_num_terms__(), self.numVotes_), 0.0, dtype=np.float32)


    def __get_num_terms__(self):
        """Method to get the number of terms that are predicted.

        Returns:
            (int): Number of terms.
        """

        return self.testDS_.get_num_functions()


    def __create_model__(self):
        """Method to create the model.
        """

        #Get the number of terms.
        numTerms = self.__get_num_terms__()

        #Create the model object.
        self.model_ = ProtClass(self.config_._sections['ProtClass'], 
            3, self.batchSize_, numTerms, self.aminoInput_)
        
        #Create the placeholders.
        if not self.aminoInput_:
            self.numInFeatures_ = self.model_.create_placeholders(3)
        else:
            self.numInFeatures_ = self.model_.create_placeholders(0)

        #Create the model.
        self.model_.create_model(0, 1)


    def __create_savers__(self):
        """Method to create the saver objects.
        """
        self.saver_ = tf.train.Saver()


    def __load_parameters__(self):
        """Method to load the parameters of a model.
        """

        #Restore the model
        self.saver_.restore(self.sess_, self.logFolder_+"/"+self.checkPointName_)


    def __test_one_voting__(self, pNumVote):
        """Private method to test on voting step.

        Args:
            pNumVote (int): Current number of vote.
        """

        numClasses = self.__get_num_terms__()

        #Calculate num batches.
        numProteins = self.testDS_.get_num_proteins()
        numBatchesTest = numProteins//self.batchSize_

        # Check if the number of proteins is not multiple of the batch size.
        if numProteins%self.batchSize_ != 0:
            numBatchesTest +=1

        #Init dataset epoch.
        self.testDS_.start_epoch()

        #Test the model.
        accuracyCats = np.full((numClasses), 0.0, dtype=np.float)
        numObjCats = np.full((numClasses), 0.0, dtype=np.float)
        for curBatch in range(numBatchesTest):
            
            #Calculate the current batch size.
            curBatchSize = min(self.batchSize_, len(self.testDS_.data_) - self.testDS_.iterator_)

            #Get the batch data.
            protBatch, features, labels = self.testDS_.get_next_batch(
                curBatchSize, self.augment_)

            #Create the dictionary for tensorflow.
            curDict = {}
            self.model_.associate_inputs_to_ph(curDict, protBatch, features, labels, False)

            #Execute a training step.
            curLogits = self.sess_.run(self.model_.logits_, curDict)

            #Incremental average.
            labels = np.argmax(labels, axis=-1) 
            for curModel in range(curBatchSize):
                #Accumulate the logits.
                curId = curBatch*self.batchSize_ + curModel
                self.accumLogits_[curId] += (curLogits[curModel] - \
                    self.accumLogits_[curId])/float(pNumVote+1)

                #Compute accuracies. 
                maxIndexs = np.argpartition(curLogits[curModel], -self.topK_)[-self.topK_:]
                if labels[curModel] in maxIndexs:
                    self.votesXClassAcc_[labels[curModel], pNumVote] += 100.0

                maxIndexs = np.argpartition(self.accumLogits_[curId], -self.topK_)[-self.topK_:]
                if labels[curModel] in maxIndexs:
                    accuracyCats[labels[curModel]] += 100.0
                numObjCats[labels[curModel]] += 1.0          

            if curBatch% 10 == 0 and curBatch > 0:
                visualize_progress(curBatch, numBatchesTest, pSameLine = True)

        #Print the result of the test.            
        totalAccuracy = np.sum(accuracyCats)/float(numProteins)
        totalAccuracyNV = np.sum(self.votesXClassAcc_[:, pNumVote])/float(numProteins)
        accuracyCatsNV = np.full((numClasses), 0.0, dtype=np.float)
        for i in range(numClasses): 
            accuracyCats[i] = accuracyCats[i]/numObjCats[i]
            accuracyCatsNV[i] = self.votesXClassAcc_[i, pNumVote]/numObjCats[i]
        print()
        print("End test")
        print("NV -> Accuracy: %.4f | Per Class Accuracy: %.4f" % 
            (totalAccuracyNV, np.mean(accuracyCatsNV)))
        print("V  -> Accuracy: %.4f | Per Class Accuracy: %.4f" % 
            (totalAccuracy, np.mean(accuracyCats)))


    def __test_aggregation__(self):
        """Private method to aggregate the results of all votes.
        """
        
        numClasses = self.__get_num_terms__()

        #Compute the accuracy.
        numProteins = self.testDS_.get_num_proteins()
        accuracyCats = np.full((numClasses), 0.0, dtype=np.float)
        numObjCats = np.full((numClasses), 0.0, dtype=np.float)
        confMatrix = np.full((numClasses, numClasses), 0.0, dtype=np.float)
        for curProtein in range(numProteins):
            curLabel = self.testDS_.dataFunctions_[curProtein]
            predLabel = np.argmax(self.accumLogits_[curProtein])
            maxIndexs = np.argpartition(self.accumLogits_[curProtein], -self.topK_)[-self.topK_:]
            if curLabel in maxIndexs:
                accuracyCats[curLabel] += 100.0
            confMatrix[curLabel, predLabel] += 1.0
            numObjCats[curLabel] += 1.0
            
        #Print the result of the test.
        totalAccuracy = np.sum(accuracyCats)/float(numProteins)
        totalAccuracyNV = np.mean(np.sum(self.votesXClassAcc_, axis=0)/float(numProteins))
        for i in range(numClasses): 
            self.votesXClassAcc_[i, :]  = self.votesXClassAcc_[i, :]/numObjCats[i]
            accuracyCats[i] = accuracyCats[i]/numObjCats[i]
        
        print("")
        for i in range(numClasses):
            print("Category %12s (%4d) ->  NV: %.4f | V: %.4f" %(
                self.testDS_.functions_[i], 
                int(numObjCats[i]), 
                np.mean(self.votesXClassAcc_[i, :]),
                accuracyCats[i]))
        print("")

        print("NV -> Accuracy: %.4f | Mean Class Accuracy: %.4f" % (totalAccuracyNV, np.mean(self.votesXClassAcc_)))
        print("V  -> Accuracy: %.4f | Mean Class Accuracy: %.4f" % (totalAccuracy, np.mean(accuracyCats)))

        #Save the confusion matrix.
        with open(self.logFolder_+"/"+self.checkPointName_+"_conf_matrix.txt", 'w') as confMatFile:
            for curClass in range(numClasses):
                for curPredClass in range(numClasses):
                    confMatFile.write(str(confMatrix[curClass, curPredClass])+";")
                confMatFile.write("\n")

        #Save the accuracies of each vote.
        with open(self.logFolder_+"/"+self.checkPointName_+"_vote_acc.txt", 'w') as voteAcc:
            for curVote in range(self.numVotes_):
                voteAcc.write(str(np.mean(self.votesXClassAcc_[:, curVote]))+";")
            voteAcc.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test protein function prediction')
    parser.add_argument('--configFile', default='test.ini', help='Configuration file (default: test.ini)')
    args = parser.parse_args()

    trainObj = ProtFunctTestLoop(args.configFile)
    trainObj.test()
