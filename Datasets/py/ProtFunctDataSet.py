'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ProtFunctdataSet.py

    \brief Dataset for the task of enzyme classification.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import h5py
import numpy as np
import copy
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.dirname(BASE_DIR)
ROOT_PROJ_DIR = os.path.dirname(TASKS_DIR)
sys.path.append(ROOT_PROJ_DIR)

from IEProtLib.py_utils.py_pc import rotate_pc_3d
from IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein, PyProteinBatch

class ProtFunctDataSet:
    """ProtFunct dataset class.
    """

    def __init__(self, pDataSet = "Training", pPath="../data/ProtFunct/", 
        pAminoInput = False, pRandSeed = None, pPermute = True, pLoadText = False,
        pAminoPool = "spec_clust"):
        """Constructor.
        """

        self.loadText_ = pLoadText
        self.aminoInput_ = pAminoInput

        self.randomState_ = np.random.RandomState(pRandSeed)

        # Save the dataset path.
        self.path_ = pPath

        # Load the file with the list of functions.
        self.functions_ = []
        with open(pPath+"/unique_functions.txt", 'r') as mFile:
            for line in mFile:
                self.functions_.append(line.rstrip())

        # Create the periodic table.
        self.periodicTable_ = PyPeriodicTable()

        # Get the file list.
        if pDataSet == "Training":
            splitFile = "/training.txt"
        elif pDataSet == "Validation":
            splitFile = "/validation.txt"
        elif pDataSet == "Test":
            splitFile = "/testing.txt"
        self.proteinNames_ = []
        self.fileList_ = []
        with open(pPath+splitFile, 'r') as mFile:
            for line in mFile:
                self.proteinNames_.append(line.rstrip())
                self.fileList_.append(pPath+"/data/"+line.rstrip())

        # Load the functions.
        print("Reading protein functions")
        self.protFunct_ = {}
        with open(pPath+"/chain_functions.txt", 'r') as mFile:
            for line in mFile:
                splitLine = line.rstrip().split(',')
                if splitLine[0] in self.proteinNames_: 
                    self.protFunct_[splitLine[0]] = int(splitLine[1])

        # Create the folder for the poolings.
        poolingMethod = pAminoPool
        if poolingMethod == "rosetta_cen":
            poolingFolder = "poolings_rosetta"
        elif poolingMethod == "spec_clust":
            poolingFolder = "poolings"
        if not os.path.exists(pPath+"/"+poolingFolder): os.mkdir(pPath+"/"+poolingFolder)

        # Load the dataset
        print("Reading the data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graphCache = {}
            self.data_ = []
            self.dataFunctions_ = []
            for fileIter, curFile in enumerate(self.fileList_):

                fileName = curFile.split('/')[-1]
                if fileIter%250 == 0:
                    print("\r# Reading file "+fileName+" ("+str(fileIter)+" of "+\
                        str(len(self.fileList_))+")", end="")
                
                curProtein = PyProtein(self.periodicTable_)
                curProtein.load_hdf5(curFile+".hdf5",
                    pLoadAtom = not self.aminoInput_, pLoadAmino = True, pLoadText = pLoadText or poolingMethod == "rosetta_cen")
                
                if os.path.exists(pPath+"/"+poolingFolder+"/"+fileName+".hdf5"):
                    curProtein.load_pooling_hdf5(pPath+"/"+poolingFolder+"/"+fileName+".hdf5")
                else:
                    curProtein.create_pooling(graphCache, poolingMethod)
                    curProtein.save_pooling_hdf5(pPath+"/"+poolingFolder+"/"+fileName+".hdf5")

                self.data_.append(curProtein)
                self.dataFunctions_.append(self.protFunct_[self.proteinNames_[fileIter]])
        print()        

        # Compute function weights.
        print("Computing function weights")
        auxFunctCount = np.full((len(self.functions_)), 0, dtype=np.int32)
        for curProt in self.proteinNames_:
            auxFunctCount[self.protFunct_[curProt]] += 1
        functProbs = auxFunctCount.astype(np.float32)/(float(len(self.proteinNames_))*0.5)
        print("Min occurence: ", np.amin(functProbs))
        print("Max occurence: ", np.amax(functProbs))
        self.functWeights_ = 1.0/functProbs
        self.functWeightsLog_ = 1.0/np.log(1.2 + functProbs)
        print()

        # Iterator. 
        self.permute_ = pPermute
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.data_))
        else:
            self.randList_ = np.arange(len(self.data_))

        # Initialize the updater iterator.
        self.updaterIter_ = 0
        
        
    def get_num_proteins(self):
        """Method to get the number of proteins in the dataset.

        Return:
            (int): Number of proteins.
        """
        return len(self.data_)

    def get_num_functions(self):
        """Method to get the number of different functions in the dataset.

        Return:
            (int): Number of functions.
        """
        return len(self.functions_)


    def start_epoch(self):
        """Method to start a new epoch.
        """
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.data_))
        else:
            self.randList_ = np.arange(len(self.data_))

    def end_epoch(self, pBatchSize):
        """Method to consult if the epoch has finished.

        Args:
            pBatchSize (int): Size of the batch.
        Returns:
            (bool): True if the epoch has finished.
        """
        return self.iterator_+pBatchSize >= len(self.data_)


    def get_next_batch(self, pBatchSize, pAugment = False):
        """Method to get the next batch. If there are not enough proteins to fill
            the batch, None is returned.

        Args:
            pBatchSize (int): Size of the batch.
            pAugment (bool): Boolean that indicates the data has to be augmented.
        Returns:
            (MCPyProteinBatch): Output protein batch.
            (float np.array n): Output features.
            (int np.array b): List of labels.
        """

        #Check for the validity of the input parameters.
        if pBatchSize <= 0:
            raise RuntimeError('Only a positive batch size is allowed.')

        #Create the output buffers.
        proteinList = []
        atomFeatures = []
        protLabels = []

        #Get the data.
        for curIter in range(pBatchSize):

            curProtein = None
            
            #If there are enough models left.
            if self.iterator_+curIter < len(self.data_):

                #Select the model.
                curProtIndex = self.randList_[self.iterator_+curIter]
                curProtein = copy.deepcopy(self.data_[curProtIndex])
                        
                #Save the augmented model.
                proteinList.append(curProtein)

                #Create the feature list.
                if not self.aminoInput_:
                    curFeatures = np.concatenate((
                        curProtein.periodicTable_.covRadius_[curProtein.atomTypes_].reshape((-1,1)),
                        curProtein.periodicTable_.vdwRadius_[curProtein.atomTypes_].reshape((-1,1)),
                        curProtein.periodicTable_.mass_[curProtein.atomTypes_].reshape((-1,1))),
                        axis=1)
                    atomFeatures.append(curFeatures)

                #Append the current label.
                probs = np.full((len(self.functions_)), 0.0, dtype=np.float32)
                probs[self.dataFunctions_[curProtIndex]] = 1.0
                protLabels.append(probs)
                
        #Increment iterator.
        self.iterator_ += pBatchSize

        #Prepare the output of the function.
        protBatch = PyProteinBatch(proteinList, self.aminoInput_, self.loadText_)
        if not self.aminoInput_:
            atomFeatures = np.concatenate(atomFeatures, axis=0)

        #Return the current batch.
        return protBatch, atomFeatures, protLabels
