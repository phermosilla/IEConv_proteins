'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyProteinBatch.py

    \brief Object to agrupate a set of proteins.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import numpy as np
import h5py

class PyProteinBatch:
    """Class to store a protein batch.
    """

    def __init__(self, pProteinList, pAminoInput = False, pLoadText = False):
        """Constructor.

        Args:
            pProteinList (PyProtein list): List of proteins.
            pAminoInput (bool): Boolean that indicates if the inputs
                are aminoacids.
        """

        # Save if the input is at aminoacid level.
        self.aminoInput_ = pAminoInput


        if self.aminoInput_:

            # Atom lists.
            self.atomPos_ = []
            self.atomTypes_ = []
            self.atomBatchIds_ = []
            self.centers_ = []

            # Pooling lists.
            self.graph1Neighs_ = [[]]
            self.graph1NeighsStartIds_ = [[]]
            self.graph2Neighs_ = [[]]
            self.graph2NeighsStartIds_ = [[]]
            curAtomIndexOffset = [0]
            curGraph1NeighOffset = [0]
            curGraph2NeighOffset = [0]

            self.poolingIds_ = None    
            self.atomAminoIds_ = None
            self.atomResidueIds_ = None
            self.numResidues_ = 0
            
            #Iterate over the protein list.
            for protIter, curProtein in enumerate(pProteinList):
            
                #Get the aminoacid information.
                self.atomPos_.append(curProtein.aminoPos_[0])
                self.atomTypes_.append(curProtein.aminoType_)
                self.atomBatchIds_.append(np.full(curProtein.aminoType_.shape, protIter, dtype=np.int32))

                #Get the neighborhood information.
                self.graph1Neighs_[0].append(curProtein.aminoNeighs_ + curAtomIndexOffset[0])
                self.graph1NeighsStartIds_[0].append(curProtein.aminoNeighsSIndices_ + curGraph1NeighOffset[0])
                self.graph2Neighs_[0].append(curProtein.aminoNeighsHB_ + curAtomIndexOffset[0])
                self.graph2NeighsStartIds_[0].append(curProtein.aminoNeighsSIndicesHB_ + curGraph2NeighOffset[0])

                #Update offsets.
                curAtomIndexOffset[0] += len(curProtein.aminoPos_[0])
                curGraph1NeighOffset[0] += len(curProtein.aminoNeighs_)
                curGraph2NeighOffset[0] += len(curProtein.aminoNeighsHB_)
            
                #Get the protein centers.
                self.centers_.append(curProtein.center_)

            #Create the numpy arrays.
            self.atomPos_ = np.concatenate(self.atomPos_, axis=0)
            self.atomTypes_ = np.concatenate(self.atomTypes_, axis=0)
            self.atomBatchIds_ = np.concatenate(self.atomBatchIds_, axis=0)
            self.centers_ = np.concatenate(self.centers_, axis=0)

            self.graph1Neighs_[0] = np.concatenate(self.graph1Neighs_[0], axis=0)
            self.graph1NeighsStartIds_[0] = np.concatenate(self.graph1NeighsStartIds_[0], axis=0)
            self.graph2Neighs_[0] = np.concatenate(self.graph2Neighs_[0], axis=0)
            self.graph2NeighsStartIds_[0] = np.concatenate(self.graph2NeighsStartIds_[0], axis=0)

        else:

            # Atom lists.
            self.atomPos_ = []
            self.aminoPos_ = []
            self.atomTypes_ = []
            self.atomBatchIds_ = []
            self.atomAminoIds_ = []
            self.atomResidueIds_ = []
            self.numResidues_ = 0
            curAminoIndexOffset = 0
            curResidueIndexOffset = 0
            self.centers_ = []

            if pLoadText:
                self.atomNames_ = []
                self.atomResidueNames_ = []

            # Pooling lists.
            startIndex = 0
            validIndex = len(np.unique(pProteinList[0].atomTypes_)) > 1
            while not validIndex:
                startIndex += 1
                validIndex = len(np.unique(pProteinList[startIndex].atomTypes_)) > 1

            self.poolingIds_ = [[] for i in range(len(pProteinList[startIndex].poolStartNeighs_))]
            self.graph1Neighs_ = [[] for i in range(len(pProteinList[startIndex].poolStartNeighs_)+2)]
            self.graph1NeighsStartIds_ = [[] for i in range(len(pProteinList[startIndex].poolStartNeighs_)+2)]
            self.graph2Neighs_ = [[] for i in range(len(pProteinList[startIndex].poolStartNeighs_)+2)]
            self.graph2NeighsStartIds_ = [[] for i in range(len(pProteinList[startIndex].poolStartNeighs_)+2)]
            curAtomIndexOffset = np.full((len(pProteinList[startIndex].poolStartNeighs_)+2), 0, dtype=np.int32)
            curGraph1NeighOffset = np.full((len(pProteinList[startIndex].poolStartNeighs_)+2), 0, dtype=np.int32)
            curGraph2NeighOffset = np.full((len(pProteinList[startIndex].poolStartNeighs_)+2), 0, dtype=np.int32)        
            
            #Iterate over the protein list.
            for protIter, curProtein in enumerate(pProteinList):

                if len(np.unique(curProtein.atomTypes_)) > 1:
                    #Process the atom aminoacid indices to keep the negative indices (atoms that do not belong to an aminoacid).
                    auxAtomAminoIds = np.array([val + curAminoIndexOffset if val >= 0 else -1 for val in curProtein.atomAminoIds_])

                    #Get the atom information.
                    self.atomPos_.append(curProtein.atomPos_[0])
                    self.aminoPos_.append(curProtein.aminoPos_[0])
                    self.atomAminoIds_.append(auxAtomAminoIds)
                    self.atomResidueIds_.append(curProtein.atomResidueIds_ + curResidueIndexOffset)
                    self.atomTypes_.append(curProtein.atomTypes_)
                    self.atomBatchIds_.append(np.full(curProtein.atomAminoIds_.shape, protIter, dtype=np.int32))
                    if pLoadText:
                        self.atomNames_.append(curProtein.atomNames_)
                        self.atomResidueNames_.append(curProtein.atomResidueNames_)

                    #Get the neighborhood information.
                    self.graph1Neighs_[0].append(curProtein.covBondList_ + curAtomIndexOffset[0])
                    self.graph1NeighsStartIds_[0].append(curProtein.atomCovBondSIndices_ + curGraph1NeighOffset[0])
                    self.graph2Neighs_[0].append(curProtein.covBondListHB_ + curAtomIndexOffset[0])
                    self.graph2NeighsStartIds_[0].append(curProtein.atomCovBondSIndicesHB_ + curGraph2NeighOffset[0])

                    #Update offsets.
                    curAtomIndexOffset[0] += len(curProtein.atomPos_[0])
                    curResidueIndexOffset += len(np.unique(curProtein.atomResidueIds_))
                    uniqueAminoIds = np.unique(curProtein.atomAminoIds_)
                    curAminoIndexOffset += len(uniqueAminoIds[uniqueAminoIds >= 0])
                    curGraph1NeighOffset[0] += len(curProtein.covBondList_)
                    curGraph2NeighOffset[0] += len(curProtein.covBondListHB_)

                    #Get the pooling operations.
                    for curPool in range(len(curProtein.poolIds_)):

                        #Get the pooling ids.
                        self.poolingIds_[curPool].append(curProtein.poolIds_[curPool] + curAtomIndexOffset[curPool+1])

                        #Get the pooled graphs.
                        if curProtein.poolNeighs_[curPool].shape[0] > 0:
                            self.graph1Neighs_[curPool+1].append(curProtein.poolNeighs_[curPool] + curAtomIndexOffset[curPool+1])
                        self.graph1NeighsStartIds_[curPool+1].append(curProtein.poolStartNeighs_[curPool] + curGraph1NeighOffset[curPool+1])
                        if curProtein.poolNeighsHB_[curPool].shape[0] > 0:    
                            self.graph2Neighs_[curPool+1].append(curProtein.poolNeighsHB_[curPool] + curAtomIndexOffset[curPool+1])
                        self.graph2NeighsStartIds_[curPool+1].append(curProtein.poolStartNeighsHB_[curPool] + curGraph2NeighOffset[curPool+1])

                        #Update offset.
                        curAtomIndexOffset[curPool+1] += len(curProtein.poolStartNeighs_[curPool])
                        curGraph1NeighOffset[curPool+1] += len(curProtein.poolNeighs_[curPool])
                        curGraph2NeighOffset[curPool+1] += len(curProtein.poolNeighsHB_[curPool])

                    #Get the amino neighs list.
                    self.graph1Neighs_[-1].append(curProtein.aminoNeighs_ + curAtomIndexOffset[-1])
                    self.graph1NeighsStartIds_[-1].append(curProtein.aminoNeighsSIndices_ + curGraph1NeighOffset[-1])
                    self.graph2Neighs_[-1].append(curProtein.aminoNeighsHB_ + curAtomIndexOffset[-1])
                    self.graph2NeighsStartIds_[-1].append(curProtein.aminoNeighsSIndicesHB_ + curGraph2NeighOffset[-1])
                    curAtomIndexOffset[-1] += len(uniqueAminoIds[uniqueAminoIds >= 0])
                    curGraph1NeighOffset[-1] += len(curProtein.aminoNeighs_)
                    curGraph2NeighOffset[-1] += len(curProtein.aminoNeighsHB_)

                    #Get the protein centers.
                    self.centers_.append(curProtein.center_)

                else:
                    self.centers_.append(np.array([0.0, 0.0, 0.0]))

            #Create the numpy arrays.
            self.atomPos_ = np.concatenate(self.atomPos_, axis=0)
            self.atomTypes_ = np.concatenate(self.atomTypes_, axis=0)
            self.atomBatchIds_ = np.concatenate(self.atomBatchIds_, axis=0)
            self.centers_ = np.concatenate(self.centers_, axis=0)
            self.aminoPos_ = np.concatenate(self.aminoPos_, axis=0)
            self.atomAminoIds_ = np.concatenate(self.atomAminoIds_, axis=0)
            self.atomResidueIds_ = np.concatenate(self.atomResidueIds_, axis=0)
            self.numResidues_ = curResidueIndexOffset 
            if pLoadText:
                self.atomNames_ = np.concatenate(self.atomNames_, axis=0)  
                self.atomResidueNames_ = np.concatenate(self.atomResidueNames_, axis=0)             

            for curPool in range(len(self.poolingIds_)):
                self.poolingIds_[curPool] =  np.concatenate(self.poolingIds_[curPool], axis=0)
            for curPool in range(len(self.graph1Neighs_)):
                self.graph1Neighs_[curPool] = np.concatenate(self.graph1Neighs_[curPool], axis=0)
                self.graph1NeighsStartIds_[curPool] = np.concatenate(self.graph1NeighsStartIds_[curPool], axis=0)
                self.graph2Neighs_[curPool] = np.concatenate(self.graph2Neighs_[curPool], axis=0)
                self.graph2NeighsStartIds_[curPool] = np.concatenate(self.graph2NeighsStartIds_[curPool], axis=0)

