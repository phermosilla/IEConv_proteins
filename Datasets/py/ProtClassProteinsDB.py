'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ProtClassProteinsDB.py

    \brief Dataset for the task of classification of enzymes vs non-enzymes.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import h5py
import copy
import warnings
import numpy as np

from IEProtLib.py_utils import visualize_progress
from IEProtLib.py_utils.py_pc import rotate_pc_3d
from IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein, PyProteinBatch

class ProtClassProteinsDB:
    """ProteinsDB dataset class.
    """

    def __init__(self, pDataset = "Training", pPath="../data/ProteinsDB/", pFoldId = "1",
        pRandSeed = None, pPermute = True, pAmino = False, pLoadText = False):
        """Constructor.
        """

        self.foldId_ = pFoldId
        self.loadText_ = pLoadText
        self.amino_ = pAmino

        # Load the fold.
        foldProteins = set()
        with open(pPath+"amino_fold_"+str(pFoldId)+".txt", 'r') as mFile:
            for curLine in mFile:
                foldProteins.add(curLine.rstrip())

        # Load the file with the list of classes.
        self.fileList_ = []
        self.annotations_ = []
        with open(pPath+"amino_enzymes.txt", 'r') as mFile:
            for curLine in mFile:
                curProtein = curLine.rstrip()
                if curProtein in foldProteins and pDataset == "Validation":
                    self.fileList_.append(pPath+"data/"+curProtein)
                    self.annotations_.append(1)
                elif not(curProtein in foldProteins) and pDataset == "Training":
                    self.fileList_.append(pPath+"data/"+curProtein)
                    self.annotations_.append(1)

        with open(pPath+"amino_no_enzymes.txt", 'r') as mFile:
            for curLine in mFile:
                curProtein = curLine.rstrip()
                if curProtein in foldProteins and pDataset == "Validation":
                    self.fileList_.append(pPath+"data/"+curProtein)
                    self.annotations_.append(0)
                elif not(curProtein in foldProteins) and pDataset == "Training":
                    self.fileList_.append(pPath+"data/"+curProtein)
                    self.annotations_.append(0)

        # Create the periodic table.
        self.periodicTable_ = PyPeriodicTable()

        # Create the folder for the poolings.
        poolingFolder = "poolings"
        if not os.path.exists(pPath+"/"+poolingFolder): os.mkdir(pPath+"/"+poolingFolder)

        # Load the dataset.
        self.onlyCAProts_ = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graphCache = {}
            self.data_ = []
            for fileIter, curFile in enumerate(self.fileList_):
                 
                fileName = curFile.split('/')[-1]
                if fileIter%100 == 0:
                    print("\r# Reading file "+fileName+" ("+str(fileIter)+" of "+\
                        str(len(self.fileList_))+")", end="")

                curProtein = PyProtein(self.periodicTable_)
                curProtein.load_hdf5(curFile+".hdf5",
                    pLoadAtom = True, pLoadAmino = True, pLoadText = pLoadText)
                
                if len(np.unique(curProtein.atomTypes_)) > 1:
                    if os.path.exists(pPath+"/"+poolingFolder+"/"+fileName+".hdf5"):
                        curProtein.load_pooling_hdf5(pPath+"/"+poolingFolder+"/"+fileName+".hdf5")
                    else:
                        curProtein.create_pooling(graphCache)
                        curProtein.save_pooling_hdf5(pPath+"/"+poolingFolder+"/"+fileName+".hdf5")
                else:
                    self.onlyCAProts_.add(fileIter)
                self.data_.append(curProtein)
        print()
        print(len(graphCache))

        # Iterator. 
        self.permute_ = pPermute
        self.randomState_ = np.random.RandomState(pRandSeed)
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.data_))
        else:
            self.randList_ = np.arange(len(self.data_))
        

    def get_amino_matrices(self):
        """Method to get the aminoacid matrices.

        Returns:
            (list of matrices): List of aminoacid matrices.
        """
        return self.aminoMatrix_


    def get_num_proteins(self):
        """Method to get the number of proteins in the dataset.

        Return:
            (int): Number of proteins.
        """
        return len(self.data_)


    def start_epoch(self):
        """Method to start a new epoch.
        """
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.data_))
        else:
            self.randList_ = np.arange(len(self.data_))


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

        # Number of valid proteins in the batch.
        validProteins = 0

        #Create the output buffers.
        proteinList = []
        atomFeatures = []
        protLabels = []
        nameList = []

        #Get the data.
        for curIter in range(pBatchSize):

            if self.iterator_+curIter < len(self.randList_):
                #Select the model.
                curProtIndex = self.randList_[self.iterator_+curIter]
                curProtein = self.data_[curProtIndex]
                curAnnotation = self.annotations_[curProtIndex]
                curName = self.fileList_[curProtIndex]

                nameList.append(curName)

                if not curProtIndex in self.onlyCAProts_:
                    validProteins += 1
                        
                #Save the augmented model.
                proteinList.append(curProtein)

                #Create the feature list.
                if not self.amino_ and not curProtIndex in self.onlyCAProts_:
                    curFeatures = np.concatenate((
                        curProtein.periodicTable_.covRadius_[curProtein.atomTypes_].reshape((-1,1)),
                        curProtein.periodicTable_.vdwRadius_[curProtein.atomTypes_].reshape((-1,1)),
                        curProtein.periodicTable_.mass_[curProtein.atomTypes_].reshape((-1,1))),
                        axis=1)
                    atomFeatures.append(curFeatures)

                #Append the current label.
                protLabels.append(curAnnotation)

        #Increment iterator.
        self.iterator_ += len(protLabels)

        #Prepare the output of the function.
        protBatch = PyProteinBatch(proteinList, self.amino_, self.loadText_)
        if not self.amino_:
            atomFeatures = np.concatenate(atomFeatures, axis=0)

        #Return the current batch.
        return protBatch, atomFeatures, protLabels, nameList, validProteins

        
