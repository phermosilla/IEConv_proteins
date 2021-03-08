'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyMoleculeBatch.py

    \brief Object to agregate a set of molecules.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import numpy as np
import h5py

class PyMoleculeBatch:
    """Class to store a molecule batch.

    Attributes:
        atomPos_ (float np.array nxd): List of atom positions.
        atomNeighs_  (int np.array mx2): List of neighbors.
        atomNeighsStartIds_ (int np.array n): List of starting indices in
            the neighbor list.
        atomBatchIds_ (int np.array n): List of batch ids.
    """

    def __init__(self, pMoleculeList):
        """Constructor.

        Args:
            pMoleculeList (MCPyMolecule list): List of molecule.
        """

        #Define the lists.
        self.atomPos_ = []
        self.atomNeighs_ = []
        self.atomNeighsStartIds_ = []
        self.atomTypes_ = []
        self.atomBatchIds_ = []
        self.centers_ = []

        #Iterate over the molecule list.
        curAtomIndexOffset = 0
        curAtomNeighOffset = 0
        for molIter, curMolecule in enumerate(pMoleculeList):
            
            #Get the current arrays.
            self.atomPos_.append(curMolecule.atomPos_[0])
            self.atomNeighs_.append(curMolecule.covBondList_ + curAtomIndexOffset)
            self.atomNeighsStartIds_.append(curMolecule.atomCovBondSIndices_ + curAtomNeighOffset)
            self.atomTypes_.append(curMolecule.atomTypes_)
            self.atomBatchIds_.append(np.full(curMolecule.atomTypes_.shape, molIter, dtype=np.int32))
            self.centers_.append(curMolecule.center_)

            #Update the offsets.
            curAtomIndexOffset += len(curMolecule.atomPos_[0])
            curAtomNeighOffset += len(curMolecule.covBondList_)

        #Concatenate the arrays.
        self.atomPos_ = np.concatenate(self.atomPos_, axis=0)
        self.atomNeighs_ = np.concatenate(self.atomNeighs_, axis=0)
        self.atomNeighsStartIds_ = np.concatenate(self.atomNeighsStartIds_, axis=0)
        self.atomTypes_ = np.concatenate(self.atomTypes_, axis=0)
        self.atomBatchIds_ = np.concatenate(self.atomBatchIds_, axis=0)