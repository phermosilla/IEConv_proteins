'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Test.py

    \brief Code to test a classification network on the Homology task.

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
from Datasets import ProtClassHomologyDataSet

current_milli_time = lambda: time.time() * 1000.0

class ProtClassTestLoop(TestLoop):
    """Class to test a classification network on the protclass100 dataset.
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
        trainConfigDict = self.config_._sections['ProtHomology']
        self.batchSize_ = int(trainConfigDict['batchsize'])
        self.augment_ = trainConfigDict['augment'] == "True"
        self.checkPointName_ = trainConfigDict['checkpointname']
        self.logFolder_ = trainConfigDict['logfolder']
        self.aminoInput_ = trainConfigDict['aminoinput'] == "True"
        self.topK_ = int(trainConfigDict['topk'])
        self.dataset_ = trainConfigDict['dataset']

        #Call the constructor of the parent.
        TestLoop.__init__(self, self.config_._sections['TestLoop'])


    def __create_datasets__(self):
        """Method to create the datasets.
        """

        print("")
        print("########## Loading test dataset")
        self.testDS_ = ProtClassHomologyDataSet(
            pDataset = self.dataset_, 
            pPath = "../../Datasets/data/HomologyTAPE",
            pRandSeed=33, 
            pPermute=False,
            pAmino = self.aminoInput_,
            pLoadText = False,)
        print(self.testDS_.get_num_proteins(), "proteins loaded")

        # Create the accumulative logits.
        self.accumLogits_ = np.full((self.testDS_.get_num_proteins(), 1195), 0.0, dtype=np.float64)

        #Declare the array to store the accuracy of each class each iteration.
        self.votesXClassAcc_ = np.full((len(self.testDS_.classes_), self.numVotes_), 0.0, dtype=np.float32)


    def __create_model__(self):
        """Method to create the model.
        """

        #Create the model object.
        self.model_ = ProtClass(self.config_._sections['ProtClass'], 
            3, self.batchSize_, 1195, self.aminoInput_)
        
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

        #Calculate num batches.
        numProteins = self.testDS_.get_num_proteins()
        numBatchesTest = numProteins//self.batchSize_

        # Check if the number of proteins is not multiple of the batch size.
        if numProteins%self.batchSize_ != 0:
            numBatchesTest +=1

        #Init dataset epoch.
        self.testDS_.start_epoch()

        #Test the model.
        accuracyCats = np.full((len(self.testDS_.classes_)), 0.0, dtype=np.float)
        numObjCats = np.full((len(self.testDS_.classes_)), 0.0, dtype=np.float)
        for curBatch in range(numBatchesTest):
            
            #Calculate the current batch size.
            curBatchSize = min(self.batchSize_, len(self.testDS_.data_) - self.testDS_.iterator_)

            #Get the batch data.
            protBatch, features, labels, _, validProts = self.testDS_.get_next_batch(
                curBatchSize, self.augment_)

            if validProts > 0:

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
            else:

                #Incremental average.
                labels = np.argmax(labels, axis=-1) 
                for curModel in range(curBatchSize):
                    numObjCats[labels[curModel]] += 1.0         

            if curBatch% 10 == 0 and curBatch > 0:
                visualize_progress(curBatch, numBatchesTest, pSameLine = True)

        #Print the result of the test.            
        totalAccuracy = np.sum(accuracyCats)/float(numProteins)
        totalAccuracyNV = np.sum(self.votesXClassAcc_[:, pNumVote])/float(numProteins)
        accuracyCatsNV = np.full((len(self.testDS_.classes_)), 0.0, dtype=np.float)
        countNonZeros = 0
        for i in range(len(self.testDS_.classes_)): 
            if numObjCats[i] > 0.0:
                accuracyCats[i] = accuracyCats[i]/numObjCats[i]
                accuracyCatsNV[i] = self.votesXClassAcc_[i, pNumVote]/numObjCats[i]
                countNonZeros += 1
            else:
                accuracyCats[i] = 0.0
                accuracyCatsNV[i] = 0.0

        print()
        print("End test")
        print("NV -> Accuracy: %.4f | Per Class Accuracy: %.4f" % 
            (totalAccuracyNV, np.sum(accuracyCatsNV)/float(countNonZeros)))
        print("V  -> Accuracy: %.4f | Per Class Accuracy: %.4f" % 
            (totalAccuracy, np.sum(accuracyCats)/float(countNonZeros)))


    def __test_aggregation__(self):
        """Private method to aggregate the results of all votes.
        """
        
        #Compute the accuracy.
        numProteins = self.testDS_.get_num_proteins()
        numClasses = len(self.testDS_.classes_)
        accuracyCats = np.full((numClasses), 0.0, dtype=np.float)
        numObjCats = np.full((numClasses), 0.0, dtype=np.float)
        confMatrix = np.full((numClasses, numClasses), 0.0, dtype=np.float)
        for curProtein in range(numProteins):
            curLabel = self.testDS_.cathegories_[curProtein]
            predLabel = np.argmax(self.accumLogits_[curProtein])
            maxIndexs = np.argpartition(self.accumLogits_[curProtein], -self.topK_)[-self.topK_:]
            if curLabel in maxIndexs:
                accuracyCats[curLabel] += 100.0
            confMatrix[curLabel, predLabel] += 1.0
            numObjCats[curLabel] += 1.0
            
        #Print the result of the test.
        totalAccuracy = np.sum(accuracyCats)/float(numProteins)
        totalAccuracyNV = np.mean(np.sum(self.votesXClassAcc_, axis=0)/float(numProteins))
        countNonZero = 0
        for i in range(len(self.testDS_.classes_)): 
            if numObjCats[i] > 0:
                self.votesXClassAcc_[i, :]  = self.votesXClassAcc_[i, :]/numObjCats[i]
                accuracyCats[i] = accuracyCats[i]/numObjCats[i]
                countNonZero += 1
            else:
                self.votesXClassAcc_[i, :]  = 0.0
                accuracyCats[i] = 0.0
        
        print("")
        for i in range(len(self.testDS_.classes_)):
            if int(numObjCats[i]) > 0:
                print("Category %6s (%4d) ->  NV: %.4f | V: %.4f" %(
                    self.testDS_.classesList_[i], 
                    int(numObjCats[i]), 
                    np.mean(self.votesXClassAcc_[i, :]),
                    accuracyCats[i]))
        print("")

        print("NV -> Accuracy: %.4f | Mean Class Accuracy: %.4f" % (totalAccuracyNV, np.sum(self.votesXClassAcc_)/float(countNonZero*self.numVotes_)))
        print("V  -> Accuracy: %.4f | Mean Class Accuracy: %.4f" % (totalAccuracy, np.sum(accuracyCats)/float(countNonZero)))

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
    parser = argparse.ArgumentParser(description='Script to test classification of proteins (ProtFold)')
    parser.add_argument('--configFile', default='test.ini', help='Configuration file (default: test.ini)')
    args = parser.parse_args()

    trainObj = ProtClassTestLoop(args.configFile)
    trainObj.test()
