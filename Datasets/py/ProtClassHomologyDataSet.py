'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ProtClassHomologyDataSet.py

    \brief Dataset for the task of fold classification.

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

class ProtClassHomologyDataSet:
    """ProtClass100 dataset class.
    """

    def __init__(self, pDataset = "training", pPath="../data/HomologyTAPE", 
        pRandSeed = None, pPermute = True, pAmino = False, pLoadText = False):
        """Constructor.
        """

        self.loadText_ = pLoadText
        self.amino_ = pAmino

        # Load the file with the list of classes.
        maxIndex = 0
        self.classes_ = {}
        with open(pPath+"/class_map.txt", 'r') as mFile:
            for line in mFile:
                lineList = line.rstrip().split('\t')
                self.classes_[lineList[0]] = int(lineList[1])
                maxIndex = max(maxIndex, int(lineList[1]))
        self.classesList_ = ["" for i in range(maxIndex+1)]
        for key, value in self.classes_.items():
            self.classesList_[value] = key

        # Create the periodic table.
        self.periodicTable_ = PyPeriodicTable()

        # Get the file list.
        numProtsXCat = np.full((len(self.classes_)), 0, dtype=np.int32)
        self.fileList_ = []
        self.cathegories_ = []
        with open(pPath+"/"+pDataset+".txt", 'r') as mFile:
            for curLine in mFile:
                splitLine = curLine.rstrip().split('\t')
                curClass = self.classes_[splitLine[-1]]
                self.fileList_.append(pPath+"/"+pDataset+"/"+splitLine[0])
                numProtsXCat[curClass] += 1
                self.cathegories_.append(curClass)

        # Create the folder for the poolings.
        if not self.amino_:
            poolingFolder = "poolings"
            if not os.path.exists(pPath+"/"+poolingFolder): os.mkdir(pPath+"/"+poolingFolder)
        else: 
            poolingMethod = ""

        # Load the dataset.
        self.onlyCAProts_ = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graphCache = {}
            self.data_ = []
            for fileIter, curFile in enumerate(self.fileList_):

                className = curFile.split('/')[-2]
                while len(className) < 6:
                    className = className+" " 
                fileName = curFile.split('/')[-1]
                if fileIter%100 == 0:
                    print("\r# Reading file "+fileName+" / "+className+" ("+str(fileIter)+" of "+\
                        str(len(self.fileList_))+")", end="")

                curProtein = PyProtein(self.periodicTable_)
                curProtein.load_hdf5(curFile+".hdf5",
                    pLoadAtom = not self.amino_, pLoadAmino = True, pLoadText = pLoadText)
                if not self.amino_:
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
        print("Protein with only CA: ", len(self.onlyCAProts_))
        print()

        # Compute weights.
        self.weights_ = numProtsXCat.astype(np.float32)/np.sum(numProtsXCat).astype(np.float32)
        self.weights_ = 1.0/np.log(1.2 + self.weights_)

        # Iterator. 
        self.permute_ = pPermute
        self.randomState_ = np.random.RandomState(pRandSeed)
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.data_))
        else:
            self.randList_ = np.arange(len(self.data_))


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

        #If there are enough models left.
        if self.iterator_ <= (len(self.data_)-pBatchSize):

            #Create the output buffers.
            proteinList = []
            atomFeatures = []
            protLabels = []
            classWeights = []

            #Get the data.
            for curIter in range(pBatchSize):

                #Select the model.
                curProtIndex = self.randList_[self.iterator_+curIter]
                curProtein = self.data_[curProtIndex]

                if not curProtIndex in self.onlyCAProts_:
                    validProteins += 1

                #Augment the data, rotate the protein.
                if pAugment:

                    # Copy the object before modifying it.
                    curProtein = copy.deepcopy(curProtein)

                    if self.amino_:
                        curPosVector = curProtein.aminoPos_[0]

                        # Rotation.
                        curPosVector, _ = rotate_pc_3d(self.randomState_, curPosVector)

                        # Gaussian noise.
                        curPosVector = curPosVector + \
                            np.clip(self.randomState_.normal(0.0, 0.1, curPosVector.shape), -0.3, 0.3)
                        
                        # Anisotropic scale.
                        deform = np.clip(self.randomState_.normal(0.0, 0.1, (1, 3)) + 1.0, 0.9, 1.1)
                        curPosVector = curPosVector*deform

                        curProtein.aminoPos_[0] = curPosVector
                    else:
                        curPosVector = curProtein.atomPos_[0]
                        curAminoPosVector = curProtein.aminoPos_[0]

                        # Rotation.
                        curPosVector, rotMat = rotate_pc_3d(self.randomState_, curPosVector)
                        curAminoPosVector = np.dot(curAminoPosVector, rotMat)
                        
                        # Gaussian noise.
                        curPosVector = curPosVector + \
                            np.clip(self.randomState_.normal(0.0, 0.1, curPosVector.shape), -0.3, 0.3)
                        curAminoPosVector = curAminoPosVector + \
                            np.clip(self.randomState_.normal(0.0, 0.1, curAminoPosVector.shape), -0.3, 0.3)

                        # Anisotropic scale.
                        deform = np.clip(self.randomState_.normal(0.0, 0.1, (1, 3)) + 1.0, 0.9, 1.1)
                        curPosVector = curPosVector*deform
                        curAminoPosVector = curAminoPosVector*deform

                        curProtein.atomPos_[0] = curPosVector
                        curProtein.aminoPos_[0] = curAminoPosVector
         
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
                probs = np.full((len(self.classes_)), 0.0, dtype=np.float32)
                probs[self.cathegories_[curProtIndex]] = 1.0
                protLabels.append(probs)

                #Append the class weights.
                classWeights.append(self.weights_[self.cathegories_[curProtIndex]])

            #Increment iterator.
            self.iterator_ += pBatchSize

            #Prepare the output of the function.
            protBatch = PyProteinBatch(proteinList, self.amino_, self.loadText_)
            if not self.amino_:
                atomFeatures = np.concatenate(atomFeatures, axis=0)

            #Return the current batch.
            return protBatch, atomFeatures, protLabels, classWeights, validProteins

        else:
            return None, None, None, None, 0

        
