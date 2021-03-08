'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyProtein.py

    \brief Protein object.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import copy
import numpy as np
import h5py

from sklearn.cluster import SpectralClustering

from IEProtLib.py_utils.py_mol import PyMolecule
from IEProtLib.py_utils.py_mol import load_protein_pdb, load_protein_mol2

class PyProtein(PyMolecule):
    """Class to store a protein.

    Attributes:
        periodicTable_ (PyPeriodicTable): Periodic table.
        atomPos_ (float array  kxnx3): Atom positions.
        atomNames_ (string array n): Atom names.
        atomTypes_ (int array n): Atom types.
        aminoPos_ (float array kxn'x3): Aminoacid positions.
        aminoType_ (int array n'): Aminoacid types.
        aminoIds_ (int array n'): Aminoacid ids.
        aminoChainIds_ (int array n'): Aminoacid chain ids.
        aminoNeighs_ (int array dx2): List of aminoacid neighbors. 
    """

    def __init__(self, pPeriodicTable):
        """Constructor.

        Args:
            pPeriodicTable (PyPeriodicTable): Periodic table.
        """
        PyMolecule.__init__(self, pPeriodicTable)

        # Declare the aminoacid variables.
        self.aminoPos_ = None
        self.aminoType_ = None
        self.aminoChainIds_ = None
        self.aminoNeighs_ = None
        self.aminoNeighsSIndices_ = None
        self.aminoNeighsHB_ = None
        self.aminoNeighsSIndicesHB_ = None

        # Declare additional variables per atom.
        self.atomAminoIds_ = None
        self.atomResidueIds_ = None
        self.atomChainIds_ = None
        self.atomResidueNames_ = None
        self.atomChainNames_ = None
        self.covBondListHB_ = None
        self.atomCovBondSIndicesHB_ = None
        
        # Declare variables to store the segmentation.
        self.segmentation_ = None
        
        # Pooling indices (This variables are not saved).
        self.poolIds_ = []
        self.poolStartNeighs_ = []
        self.poolNeighs_ = []
        self.poolStartNeighsHB_ = []
        self.poolNeighsHB_ = []


    def compute_min_dists_to_atoms(self, pAtomPosition):
        """Method to compute the minimum distances of the atoms of the protein
            to a set of given atoms.

        Args:
            pAtomPosition (np.array nx3): List of atom positions.
    
        Returns:
            (float numpy array): List of minimum distance of each protein atom.
        """

        transProtPos = self.atomPos_[0] + self.center_.reshape((-1, 3))
        outDists = np.full((len(self.atomNames_)), 10000.0, dtype=np.float32)
        for curAtomPos in pAtomPosition:
            dist2Atoms = np.linalg.norm(transProtPos - curAtomPos.reshape((1, 3)), axis=-1)
            outDists = np.minimum(outDists, dist2Atoms)

        return outDists


    def create_segmentation(self, pAtomPosition, pDistance= 0.0):
        """Method to create a segmentation of the protein from a list of atom positions.

        Args:
            pAtomPosition (np.array nx3): List of atom positions.
            pDistance (float): Distance that defines the radius of effect used for the 
                segmentation.
        """
        transProtPos = self.atomPos_[0] + self.center_.reshape((-1, 3))
        self.segmentation_ = np.full((len(self.atomNames_)), 0, dtype=np.int32)
        for curAtomPos in pAtomPosition:
            transCurAtomPos = curAtomPos
            dist2Atoms = np.linalg.norm(transProtPos - transCurAtomPos.reshape((1, 3)), axis=-1)
            if pDistance > 0.0:
                maskAtoms = dist2Atoms < pDistance
                self.segmentation_[maskAtoms] = 1
            else:
                minIndex = np.argmin(dist2Atoms)
                if dist2Atoms[minIndex] < 1e-9:
                    self.segmentation_[minIndex] = 1

        # Check if we found all the atoms in the protein
        if len(pAtomPosition) != np.sum(self.segmentation_) and pDistance == 0.0:  
            return False
        return True


    def get_residue_segmentation(self):
        """Method to get the segmentation at residue level.

        Return:
            (numpy array): Segmentation at residue level.
        """
        order = np.lexsort((self.segmentation_, self.atomResidueIds_))
        auxResidueIds = self.atomResidueIds_[order]
        auxSegmentation = self.segmentation_[order]
        index = np.empty(len(auxResidueIds), 'bool')
        index[-1] = True
        index[:-1] = auxResidueIds[1:] != auxResidueIds[:-1]
        return auxSegmentation[index]


    def get_aminoacid_segmentation(self):
        """Method to get the segmentation at aminoacid level.

        Return:
            (numpy array): Segmentation at aminoacid level.
        """
        order = np.lexsort((self.segmentation_, self.atomAminoIds_))
        auxResidueIds = self.atomAminoIds_[order]
        auxSegmentation = self.segmentation_[order]
        index = np.empty(len(auxResidueIds), 'bool')
        index[-1] = True
        index[:-1] = auxResidueIds[1:] != auxResidueIds[:-1]
        auxSegmentation = auxSegmentation[index]
        if auxResidueIds[0] < 0:
            return auxSegmentation[1:] #To remove aminoacid index -1 
        else:
            return auxSegmentation


    def compute_hydrogen_bonds(self):
        """Method to compute the hydrogen bonds in the protein.
        """

        # Select the backbone atoms for each aminoacid.
        caMask = np.logical_and(self.atomNames_== "CA", self.atomTypes_ == 5)
        cMask = np.logical_and(self.atomNames_== "C", self.atomAminoIds_ >= 0)
        nMask = np.logical_and(self.atomNames_== "N", self.atomAminoIds_ >= 0)
        oMask = np.logical_and(self.atomNames_== "O", self.atomAminoIds_ >= 0)
        
        caPos = self.atomPos_[0, caMask]
        cPos = self.atomPos_[0, cMask]
        nPos = self.atomPos_[0, nMask]
        oPos = self.atomPos_[0, oMask]

        nIds = np.nonzero(nMask)[0]
        oIds = np.nonzero(oMask)[0]

        if len(caPos) != len(cPos) or len(caPos) != len(nPos) or len(caPos) != len(oPos):

            caAminoIds = self.atomAminoIds_[caMask]
            cAminoIds = self.atomAminoIds_[cMask]
            nAminoIds = self.atomAminoIds_[nMask]
            oAminoIds = self.atomAminoIds_[oMask]
            cAminoIter = 0
            nAminoIter = 0
            oAminoIter = 0
            newCPos = []
            newNPos = []
            newOPos = []
            newNIds = []
            newOIds = []
            for curAminoIter, curAminoId in enumerate(caAminoIds):
                if oAminoIter < len(oAminoIds):
                    curOAminoId = oAminoIds[oAminoIter]
                else:
                    curOAminoId = -1
                if cAminoIter < len(cAminoIds):
                    curCAminoId = cAminoIds[cAminoIter]
                else:
                    curCAminoId = -1
                if nAminoIter < len(nAminoIds):
                    curNAminoId = nAminoIds[nAminoIter]
                else:
                    curNAminoId = -1

                if curAminoId == curOAminoId:
                    newOPos.append(oPos[oAminoIter])
                    newOIds.append(oIds[oAminoIter])
                    oAminoIter += 1
                else:
                    newOPos.append(caPos[curAminoIter])
                    newOIds.append(-1)
                
                if curAminoId == curCAminoId:
                    newCPos.append(cPos[cAminoIter])
                    cAminoIter += 1
                else:
                    newCPos.append(caPos[curAminoIter])

                if curAminoId == curNAminoId:
                    newNPos.append(nPos[nAminoIter])
                    newNIds.append(nIds[nAminoIter])
                    nAminoIter += 1
                else:
                    newNPos.append(caPos[curAminoIter])
                    newNIds.append(-1)

            cPos = np.array(newCPos)
            nPos = np.array(newNPos)
            oPos = np.array(newOPos)
            nIds = np.array(newNIds)
            oIds = np.array(newOIds)

        # Get the position of the previous C atom in the backbone for each aminoacid.
        cPrev = []
        oPrev = []
        for curIter in range(len(caPos)):
            startIndex = 0
            if curIter > 0:
                startIndex = self.aminoNeighsSIndices_[curIter-1]
            endIndex = self.aminoNeighsSIndices_[curIter]
            selCPos = np.array(cPos[curIter])
            selOPos = np.array(oPos[curIter])
            for curNeighIter in range(endIndex-startIndex):
                curNeigh = self.aminoNeighs_[curNeighIter + startIndex, 0]
                if curNeigh < curIter:
                    selCPos = cPos[curNeigh]
                    selOPos = oPos[curNeigh]
            cPrev.append(selCPos)
            oPrev.append(selOPos)
        
        # Compute the position of the hydrogen atom.
        cPrev = np.array(cPrev)
        oPrev = np.array(oPrev)
        prevVec = cPrev - oPrev 
        prevVec = prevVec / (np.linalg.norm(prevVec, axis = 1, keepdims=True) + 1e-9)
        hPos = nPos + prevVec

        # Compute the hydrogen bonds.
        distON = oPos.reshape((-1, 1, 3)) - nPos.reshape((1, -1, 3))
        distCH = cPos.reshape((-1, 1, 3)) - hPos.reshape((1, -1, 3))
        distOH = oPos.reshape((-1, 1, 3)) - hPos.reshape((1, -1, 3))
        distCN = cPos.reshape((-1, 1, 3)) - nPos.reshape((1, -1, 3))
        distON = np.linalg.norm(distON, axis = -1)
        distCH = np.linalg.norm(distCH, axis = -1)
        distOH = np.linalg.norm(distOH, axis = -1)
        distCN = np.linalg.norm(distCN, axis = -1)
        distON = 1.0 / (distON + 1e-9)        
        distCH = 1.0 / (distCH + 1e-9)
        distOH = 1.0 / (distOH + 1e-9)
        distCN = 1.0 / (distCN + 1e-9)

        U = (0.084 * 332.0) * (distON + distCH - distOH - distCN)
        for curIter in range(len(caPos)):
            U[curIter, curIter] = 0.0
            startIndex = 0
            if curIter > 0:
                startIndex = self.aminoNeighsSIndices_[curIter-1]
            endIndex = self.aminoNeighsSIndices_[curIter]
            for curNeighIter in range(endIndex-startIndex):
                curNeigh = self.aminoNeighs_[curNeighIter + startIndex, 0]
                U[curIter, curNeigh] = 0.0

        minIndex = np.argmin(U, axis=1)
        EMin = np.zeros_like(U)
        for i in range(U.shape[0]):
            EMin[i, minIndex[i]] = np.amin(U[i, :])
        maskHBonds = EMin < -0.5
        maskHBondsVals = maskHBonds.astype(np.int32)
        hBondsIndexs = np.transpose(np.nonzero(maskHBonds))

        # Create a new graph for the covalent bonds and hydrogen bonds together.
        self.covBondListHB_ = []
        self.atomCovBondSIndicesHB_ = []
        curStartIndex = 0
        for curIter, curEndIndex in enumerate(self.atomCovBondSIndices_):

            for curNeighbor in self.covBondList_[curStartIndex:curEndIndex]:
                self.covBondListHB_.append(curNeighbor)

            curAminoIndex = self.atomAminoIds_[curIter]
            if curAminoIndex >= 0:

                if self.atomNames_[curIter] == "O" and np.sum(maskHBondsVals[curAminoIndex, :]) > 0:
                    curAminoNeigh = np.nonzero(maskHBonds[curAminoIndex, :])[0][0]
                    if nIds[curAminoNeigh] >= 0:
                        self.covBondListHB_.append([nIds[curAminoNeigh], curIter])
                elif self.atomNames_[curIter] == "N" and np.sum(maskHBondsVals[:, curAminoIndex]) > 0:
                    curAminoNeigh = np.nonzero(maskHBonds[:, curAminoIndex])[0][0]
                    if oIds[curAminoNeigh] >= 0:
                        self.covBondListHB_.append([oIds[curAminoNeigh], curIter])

            self.atomCovBondSIndicesHB_.append(len(self.covBondListHB_))
            curStartIndex = curEndIndex

        self.covBondListHB_ = np.array(self.covBondListHB_)
        self.atomCovBondSIndicesHB_ = np.array(self.atomCovBondSIndicesHB_)

        # Create a new graph for the peptide bonds and hydrogen bonds together.
        self.aminoNeighsHB_ = []
        self.aminoNeighsSIndicesHB_ = []
        curStartIndex = 0
        for curIter, curEndIndex in enumerate(self.aminoNeighsSIndices_):

            for curNeighbor in self.aminoNeighs_[curStartIndex:curEndIndex]:
                self.aminoNeighsHB_.append(curNeighbor)

            if np.sum(maskHBondsVals[curIter, :]) > 0:
                curAminoNeigh = np.nonzero(maskHBonds[curIter, :])[0][0]
                self.aminoNeighsHB_.append([curAminoNeigh, curIter])
            if np.sum(maskHBondsVals[:, curIter]) > 0:
                curAminoNeigh = np.nonzero(maskHBonds[:, curIter])[0][0]
                self.aminoNeighsHB_.append([curAminoNeigh, curIter])

            self.aminoNeighsSIndicesHB_.append(len(self.aminoNeighsHB_))
            curStartIndex = curEndIndex

        self.aminoNeighsHB_ = np.array(self.aminoNeighsHB_)
        self.aminoNeighsSIndicesHB_ = np.array(self.aminoNeighsSIndicesHB_)


    def __update_neighborhood__(self,
        pNewIndices,
        pGraphNeighs,
        pGraphStartNeighs):

        newStartIndices = np.full((np.amax(pNewIndices)+1), 0, dtype=np.int32)

        if pGraphNeighs.shape[0] > 0:
        
            # Compute the new neighbors.
            column1 = pNewIndices[pGraphNeighs[:, 0]]
            column2 = pNewIndices[pGraphNeighs[:, 1]]
            newNeighs = np.concatenate((
                column1.reshape((-1,1)), 
                column2.reshape((-1,1))),
                axis=1)

            # Remove duplicate neighbors.
            newNeighsFiltered = []
            neighsDict = set()
            for curNeigh in newNeighs:
                if curNeigh[0] != curNeigh[1] \
                    and curNeigh[0] >= 0 and curNeigh[1] >= 0:
                    
                    keyNeigh = str(curNeigh[0])+"_"+str(curNeigh[1])
                    if not(keyNeigh in neighsDict):
                        neighsDict.add(keyNeigh)
                        newNeighsFiltered.append(curNeigh)
                        newStartIndices[curNeigh[1]] += 1
            newNeighsFiltered = np.array(newNeighsFiltered)
            if newNeighsFiltered.shape[0] > 0:
                newNeighsFiltered = newNeighsFiltered[np.argsort(newNeighsFiltered[:, 1])]

            # Compute the start indices.
            newStartIndices = np.cumsum(newStartIndices)
        else:
            newNeighsFiltered = np.array([])

        return newNeighsFiltered, newStartIndices


    def __compute_side_chain_pooling__(self, 
        pAtomAminoIds, 
        pGraphNeighs1,
        pGraphStartNeighs1,
        pGraphNeighs2,
        pGraphStartNeighs2, 
        pCacheGraph):
        
        # Reduce by half the number of atoms for each aminoacid using spectral clustering on
        # the graph defined by the covalent bonds.
        newAtomCounter = 0
        aminoCounter = 0
        newAminoIds = []
        newIndices = np.full((len(pGraphStartNeighs1)), -1, dtype=np.int32)
        uniqueAminoIds = np.unique(pAtomAminoIds)
        for curAminoId in uniqueAminoIds:
            if curAminoId >= 0:
                aminoAtomsMask = pAtomAminoIds == curAminoId
                aminoAtomsIndices = np.where(aminoAtomsMask)[0]

                if len(aminoAtomsIndices) > 1:

                    #print(len(aminoAtomsIndices), curAminoId)

                    adjMatrix = np.full((len(aminoAtomsIndices), len(aminoAtomsIndices)), 0, dtype=np.int32)

                    for curAtomIter, curAtom in enumerate(aminoAtomsIndices):
                        startNeighId = 0
                        if curAtom > 0:
                            startNeighId = pGraphStartNeighs1[curAtom-1]
                        endNeighId = pGraphStartNeighs1[curAtom]
                        for curNeigh in pGraphNeighs1[startNeighId:endNeighId]:
                            if curNeigh[0] in aminoAtomsIndices:
                                neighIter = np.where(aminoAtomsIndices == curNeigh[0])[0][0]
                                adjMatrix[curAtomIter, neighIter] = 1
                    
                    assingment = None
                    matrixKey = adjMatrix.tobytes()
                    if matrixKey in pCacheGraph:
                        assingment = pCacheGraph[matrixKey]
                    else:
                        randState = np.random.RandomState(0)
                        sc = SpectralClustering(len(aminoAtomsIndices)//2, 
                          affinity='precomputed', random_state=randState)
                        sc.fit(adjMatrix)
                        assingment = sc.labels_
                        pCacheGraph[matrixKey] = assingment

                    for curAtomIter, curAtom in enumerate(aminoAtomsIndices):
                        newIndices[curAtom] = assingment[curAtomIter] + newAtomCounter

                    for auxIter in range(len(aminoAtomsIndices)//2):
                        newAminoIds.append(aminoCounter)
                
                    aminoCounter += 1
                    newAtomCounter += len(aminoAtomsIndices)//2
                
                else:

                    newIndices[aminoAtomsIndices[0]] = newAtomCounter
                    newAminoIds.append(aminoCounter)
                    aminoCounter += 1
                    newAtomCounter += 1

        newAminoIds = np.array(newAminoIds)

        # Update the first graph.
        newNeighs1, newStartNeighs1 = self.__update_neighborhood__(
            newIndices,
            pGraphNeighs1,
            pGraphStartNeighs1)

        # Update the second graph.
        newNeighs2, newStartNeighs2 = self.__update_neighborhood__(
            newIndices,
            pGraphNeighs2,
            pGraphStartNeighs2)

        return newIndices, newNeighs1, newStartNeighs1, \
            newNeighs2, newStartNeighs2, newAminoIds

    
    def __compute_side_chain_pooling_rosetta_cen__(self, 
        pAtomNames,
        pAtomAminoIds, 
        pGraphNeighs1,
        pGraphStartNeighs1,
        pGraphNeighs2,
        pGraphStartNeighs2):
        
        # Reduce by half the number of atoms for each aminoacid using spectral clustering on
        # the graph defined by the covalent bonds.
        newAtomCounter = 0
        aminoCounter = 0
        newAminoIds = []
        newIndices = np.full((len(pGraphStartNeighs1)), -1, dtype=np.int32)
        uniqueAminoIds = np.unique(pAtomAminoIds)
        clusterCounter = 0
        for curAminoId in uniqueAminoIds:
            if curAminoId >= 0:
                aminoAtomsMask = pAtomAminoIds == curAminoId
                aminoAtomsIndices = np.where(aminoAtomsMask)[0]
            
                sideChain = -1
                clusterIds = {
                    "CA": -1,
                    "CB": -1,
                    "C" : -1,
                    "N" : -1,
                    "O" : -1}

                numNewClusters = 0
                atomClusterIds = []
                for curIndex in aminoAtomsIndices:
                    curName = pAtomNames[curIndex]
                    if curName in clusterIds:
                        if clusterIds[curName] >= 0:
                            atomClusterIds.append(clusterIds[curName])
                        else:
                            clusterIds[curName] = clusterCounter
                            atomClusterIds.append(clusterCounter)
                            clusterCounter+=1
                            numNewClusters += 1
                    else:
                        if sideChain >= 0:
                            atomClusterIds.append(sideChain)
                        else:
                            sideChain = clusterCounter
                            atomClusterIds.append(sideChain)
                            clusterCounter+=1
                            numNewClusters += 1

                for curAtomIter, curAtom in enumerate(aminoAtomsIndices):
                    newIndices[curAtom] = atomClusterIds[curAtomIter]

                for auxIter in range(numNewClusters):
                    newAminoIds.append(aminoCounter)
            
                aminoCounter += 1

        newAminoIds = np.array(newAminoIds)

        # Update the first graph.
        newNeighs1, newStartNeighs1 = self.__update_neighborhood__(
            newIndices,
            pGraphNeighs1,
            pGraphStartNeighs1)

        # Update the second graph.
        newNeighs2, newStartNeighs2 = self.__update_neighborhood__(
            newIndices,
            pGraphNeighs2,
            pGraphStartNeighs2)

        return newIndices, newNeighs1, newStartNeighs1, \
            newNeighs2, newStartNeighs2, newAminoIds


    def create_pooling(self, pCacheGraph, pMethod = "spec_clust"):
        """Method to compute the pooling indices.

        Args:
            pCacheGraph (dict): Dictionary with the pooling operation for different
                amino graphs.
            pMethod (string): Method used for clustering:
                - spec_clust: Spectral clustering.
                - rosetta_cen: Rosetta centroid.
        """
        
        # Initialize the list of pooling layers.
        self.poolIds_ = []
        self.poolStartNeighs_ = []
        self.poolNeighs_ = []
        self.poolStartNeighsHB_ = []
        self.poolNeighsHB_ = []

        # Compute the side chain pooling 1.
        if pMethod == "spec_clust":
            curPoolIds, curPoolNeighs1, curPoolStartNeighs1, \
                curPoolNeighs2, curPoolStartNeighs2, \
                curAminoIds = \
                self.__compute_side_chain_pooling__(
                    self.atomAminoIds_,
                    self.covBondList_,
                    self.atomCovBondSIndices_,
                    self.covBondListHB_,
                    self.atomCovBondSIndicesHB_, 
                    pCacheGraph)
        elif pMethod == "rosetta_cen":
            curPoolIds, curPoolNeighs1, curPoolStartNeighs1, \
                curPoolNeighs2, curPoolStartNeighs2, \
                curAminoIds = \
                self.__compute_side_chain_pooling_rosetta_cen__(
                    self.atomNames_,
                    self.atomAminoIds_,
                    self.covBondList_,
                    self.atomCovBondSIndices_,
                    self.covBondListHB_,
                    self.atomCovBondSIndicesHB_)
            
        self.poolIds_.append(curPoolIds)
        self.poolStartNeighs_.append(curPoolStartNeighs1)
        self.poolNeighs_.append(curPoolNeighs1)
        self.poolStartNeighsHB_.append(curPoolStartNeighs2)
        self.poolNeighsHB_.append(curPoolNeighs2)


    def load_molecular_file(self, pFilePath, pLoadAnim = True, pFileType = "pdb", 
        pLoadHydrogens = False, pLoadH2O = False, pBackBoneOnly = False, 
        pChainFilter = None):
        """Method to set the content of the protein from a molecular file.

        Args:
            pFilePath (string): Path to the file.
            pLoadAnim (bool): Boolean that indicates if the animation is loaded.
            pFileType (string): Type of file to load.
            pLoadHydrogens (bool): Boolean that indicates if we load the hydrogen atoms.
            pLoadH2O (bool): Boolean that indicates if we laod the water molecules.
            pBackBoneOnly (bool): Boolean that indicates if only the backbone atoms are loaded.
            pChainFilter (string): Name of the chain to filter for.
        """

        if pFileType == "pdb":
            # Load the pdb file.
            atomPos, atomTypes, atomNames, atomResidueIds, atomResidueType, \
                atomChainName, transCenter = load_protein_pdb(pFilePath, pLoadAnim,
                pLoadHydrogens = pLoadHydrogens, pLoadH2O = pLoadH2O,
                pLoadGroups=not pBackBoneOnly, pChainFilter = pChainFilter)
        elif pFileType == "mol2":
            # Load mol2 file.
            atomPos, atomTypes, atomNames, atomResidueIds, atomResidueType, \
                atomChainName, transCenter = load_protein_mol2(pFilePath, 
                pLoadHydrogens = pLoadHydrogens, pLoadH2O = pLoadH2O,
                pLoadGroups=not pBackBoneOnly, pChainFilter = pChainFilter)

        # Save the original center of the molecule.
        self.center_ = transCenter

        # Get the atom type indexs.
        auxAtomTypes = np.array([self.periodicTable_.get_atom_index(curIndex) \
            for curIndex in atomTypes])

        # Compute the mask of valid atoms.
        maskValidAtoms = auxAtomTypes >= 0
        auxAtomTypes = auxAtomTypes[maskValidAtoms]

        # Process the atom information.
        self.atomPos_ = atomPos[:, maskValidAtoms]
        self.atomNames_ = atomNames[maskValidAtoms]
        self.atomTypes_ = auxAtomTypes

        # Convert chain names to ids.
        self.atomChainNames_ = atomChainName[maskValidAtoms]
        chainNames = np.unique(self.atomChainNames_)
        self.atomChainIds_ = np.array([np.where(chainNames == curChainName)[0][0] \
            for curChainName in self.atomChainNames_])

        # Compute residue ids.
        auxResidueIds = atomResidueIds[maskValidAtoms] - np.amin(atomResidueIds[maskValidAtoms]) + 1
        auxResidueIds = auxResidueIds + self.atomChainIds_*np.amax(auxResidueIds)
        _, self.atomResidueIds_ = np.unique(auxResidueIds, return_inverse=True)
        auxResidueTypes = atomResidueType[maskValidAtoms]
        self.atomResidueNames_ = auxResidueTypes

        # Convert the aminoacid label to id.
        auxResidueTypes = np.array([self.periodicTable_.get_aminoacid_index(curIndex) \
            for curIndex in auxResidueTypes])

        # Get aminoacid information.
        mask = np.logical_and(self.atomNames_== "CA", self.atomTypes_ == 5)
        self.aminoPos_ = self.atomPos_[:, mask]
        self.aminoType_ = auxResidueTypes[mask]
        self.aminoChainIds_ = self.atomChainIds_[mask].reshape((-1))

        # Process the amino ids.
        aminoOrigIds = auxResidueIds[mask]
        self.atomAminoIds_ = np.array([np.where(aminoOrigIds == curAminoId)[0] \
            for curAminoId in auxResidueIds])
        self.atomAminoIds_ = np.array([-1 if len(curIndex)==0 else curIndex[0]
            for curIndex in self.atomAminoIds_])

        # Only load the atom belonging to the backbone.
        if pBackBoneOnly:
            bbAtomMask = self.atomAminoIds_ >= 0
            self.atomAminoIds_ = self.atomAminoIds_[bbAtomMask]
            self.atomResidueNames_ = self.atomResidueNames_[bbAtomMask]
            self.atomResidueIds_ = self.atomResidueIds_[bbAtomMask]
            _, self.atomResidueIds_ = np.unique(self.atomResidueIds_, return_inverse=True)
            self.atomPos_ = self.atomPos_[:, bbAtomMask]
            self.atomNames_ = self.atomNames_[bbAtomMask]
            self.atomTypes_ = self.atomTypes_[bbAtomMask]
            self.atomChainNames_ = self.atomChainNames_[bbAtomMask]
            self.atomChainIds_ = self.atomChainIds_[bbAtomMask]

        # Get the neighboring aminoacid information.
        self.aminoNeighs_ = []
        self.aminoNeighsSIndices_ =  np.full((len(aminoOrigIds)), 0, dtype=np.int32)
        for aminoIter in range(len(aminoOrigIds)):
            if aminoIter > 0 and \
                ((aminoOrigIds[aminoIter]-aminoOrigIds[aminoIter-1]) == 1) and \
                (self.aminoChainIds_[aminoIter] == self.aminoChainIds_[aminoIter-1]):
                self.aminoNeighs_.append([aminoIter-1, aminoIter])
            if aminoIter < len(aminoOrigIds)-1 and \
                ((aminoOrigIds[aminoIter+1]-aminoOrigIds[aminoIter]) == 1) and \
                (self.aminoChainIds_[aminoIter] == self.aminoChainIds_[aminoIter+1]):
                self.aminoNeighs_.append([aminoIter+1, aminoIter])
            self.aminoNeighsSIndices_[aminoIter] = len(self.aminoNeighs_)
        self.aminoNeighs_ = np.array(self.aminoNeighs_) 

        # Check if there is some inconsistency with the amount of aminoacids.
        if len(np.unique(self.atomAminoIds_[self.atomAminoIds_>=0])) != len(aminoOrigIds):
            print("############## ERROR!!!")
            print(pFilePath)
            print(len(np.unique(self.atomAminoIds_[self.atomAminoIds_>=0])), len(aminoOrigIds))
            print(np.unique(self.atomAminoIds_[self.atomAminoIds_>=0]))
            print(aminoOrigIds)
            print("")


    def get_fasta_seq(self):
        """Method to get the FASTA sequence of the protein.

        Returns:
            List of String: FASTA sequences of the protein.
        """

        # Correct the negative indexs.
        auxAminoType = np.array([curIndex if curIndex>=0 
            else (len(self.periodicTable_.aLetters_)-1)
            for curIndex in self.aminoType_])
    
        # Get the sequence letters.
        retSeqList = self.periodicTable_.aLetters_[auxAminoType]
        seq = "".join(retSeqList)

        # Get the different chain starts.
        prevChain = -1
        chainStarting = []
        for auxIter, curChainId in enumerate(self.aminoChainIds_):
            if prevChain != curChainId:
                chainStarting.append(auxIter)
            prevChain = curChainId
        chainStarting.append(len(self.aminoChainIds_))

        # Get the sequences of the different chains.
        retSequences = []
        for curChain in range(len(chainStarting)-1):
            chainStart = chainStarting[curChain]
            chainEnd = chainStarting[curChain+1]
            retSequences.append(seq[chainStart:chainEnd])

        return retSequences, seq

    
    def save_hdf5(self, pFilePath):
        """Method to save the protein in a hdf5 file.

        Args:
            pFilePath (string): File path.
        """
        
        h5File = h5py.File(pFilePath, "w")

        # Save atoms.
        auxAtomNames = np.array([curName.encode('utf8') for curName in self.atomNames_])
        h5File.create_dataset("pos_center", data=self.center_)
        h5File.create_dataset("atom_pos", data=self.atomPos_)
        h5File.create_dataset("atom_names", data=auxAtomNames)
        h5File.create_dataset("atom_types", data=self.atomTypes_)
        h5File.create_dataset("cov_bond_list", data=self.covBondList_)
        h5File.create_dataset("cov_bond_list_sindices", data=self.atomCovBondSIndices_)
        
        # Save additional atom info.
        auxAtomResNames = np.array([curName.encode('utf8') for curName in self.atomResidueNames_])
        auxAtomChainNames = np.array([curName.encode('utf8') for curName in self.atomChainNames_])
        h5File.create_dataset("atom_amino_id", data=self.atomAminoIds_)
        h5File.create_dataset("atom_residue_id", data=self.atomResidueIds_)
        h5File.create_dataset("atom_chain_ids", data=self.atomChainIds_)
        h5File.create_dataset("atom_residue_names", data=auxAtomResNames)
        h5File.create_dataset("atom_chain_names", data=auxAtomChainNames)
        h5File.create_dataset("cov_bond_list_hb", data=self.covBondListHB_)
        h5File.create_dataset("cov_bond_list_sindices_hb", data=self.atomCovBondSIndicesHB_)
        
        # Save aminoacids.
        h5File.create_dataset("amino_pos", data=self.aminoPos_)
        h5File.create_dataset("amino_types", data=self.aminoType_)
        h5File.create_dataset("amino_chains", data=self.aminoChainIds_)
        h5File.create_dataset("amino_neighs", data=self.aminoNeighs_)
        h5File.create_dataset("amino_neighs_sindices", data=self.aminoNeighsSIndices_)
        h5File.create_dataset("amino_neighs_hb", data=self.aminoNeighsHB_)
        h5File.create_dataset("amino_neighs_sindices_hb", data=self.aminoNeighsSIndicesHB_)

        # Save segmentation
        if not(self.segmentation_ is None):
            h5File.create_dataset("segmentation", data=self.segmentation_)

        h5File.close()


    def load_hdf5(self, pFilePath, pLoadAtom = True, pLoadAmino = True, pLoadText = True):
        """Method to load a protein from a hdf5 file.

        Args:
            pFilePath (string): File path.
            pLoadAtom (bool): Boolean that indicates if the atoms will be loaded.
            pLoadAmino (bool): Boolean that indicates if the aminoacids will be loaded.
            pLoadText (bool): Boolean that indicates if the text such as atom names, 
                residue names, or chain names will be loaded.
        """
        h5File = h5py.File(pFilePath, "r")
        
        # Load the center of the molecule.
        self.center_ = h5File["pos_center"][()]

        if pLoadAtom:

            # Load atoms.
            self.atomPos_ = h5File["atom_pos"][()]
            self.atomTypes_ = h5File["atom_types"][()]
            self.covBondList_ = h5File["cov_bond_list"][()]
            self.atomCovBondSIndices_ = h5File["cov_bond_list_sindices"][()]

            # Load atom additional info.
            self.atomAminoIds_ = h5File["atom_amino_id"][()]
            self.atomResidueIds_ = h5File["atom_residue_id"][()]
            self.atomChainIds_ = h5File["atom_chain_ids"][()]
            self.covBondListHB_ = h5File["cov_bond_list_hb"][()]
            self.atomCovBondSIndicesHB_ = h5File["cov_bond_list_sindices_hb"][()]

            # Load atom text variables.
            if pLoadText:
                auxAtomNames = h5File["atom_names"][()]
                auxAtomResNames = h5File["atom_residue_names"][()]
                auxAtomChainNames = h5File["atom_chain_names"][()]
                self.atomNames_ = np.array([curName.decode('utf8') for curName in auxAtomNames])
                self.atomResidueNames_ = np.array([curName.decode('utf8') for curName in auxAtomResNames])
                self.atomChainNames_ = np.array([curName.decode('utf8') for curName in auxAtomChainNames])
            else:
                self.atomNames_ = None
                self.atomResidueNames_ = None
                self.atomChainNames_ = None
        else:
            self.atomPos_ = None
            self.atomTypes_ = None
            self.atomNames_ = None
            self.atomResidueNames_ = None
            self.atomChainNames_ = None
            self.covBondList_ = None
            self.atomCovBondSIndices_ = None
            self.atomAminoIds_ = None
            self.atomResidueIds_ = None
            self.atomChainIds_ = None
            self.covBondListHB_ = None
            self.atomCovBondSIndicesHB_ = None

            
        if pLoadAmino:

            # Load aminoacids.
            self.aminoPos_ = h5File["amino_pos"][()]
            self.aminoType_ = h5File["amino_types"][()]
            self.aminoChainIds_ = h5File["amino_chains"][()]
            self.aminoNeighs_ = h5File["amino_neighs"][()]
            self.aminoNeighsSIndices_ = h5File["amino_neighs_sindices"][()]
            self.aminoNeighsHB_ = h5File["amino_neighs_hb"][()]
            self.aminoNeighsSIndicesHB_ = h5File["amino_neighs_sindices_hb"][()]

        else:
            self.aminoPos_ = None
            self.aminoType_ = None
            self.aminoChainIds_ = None
            self.aminoNeighs_ = None
            self.aminoNeighsSIndices_ = None
            self.aminoNeighsHB_ = None
            self.aminoNeighsSIndicesHB_ = None


        # Load segmentation.
        if "segmentation" in h5File.keys():
            self.segmentation_ = h5File["segmentation"][()]
        else:
            self.segmentation_ = None
            
        h5File.close()


    def save_pooling_hdf5(self, pFilePath):
        """Method to save the protein pooling in a hdf5 file.

        Args:
            pFilePath (string): File path.
        """

        h5File = h5py.File(pFilePath, "w")

        numPoolings = len(self.poolIds_)
        h5File.create_dataset("num_poolings", data=np.array([numPoolings]))
        for curPool in range(numPoolings):
            h5File.create_dataset("pool_ids_"+str(curPool), data=self.poolIds_[curPool])
            h5File.create_dataset("pool_neighs_"+str(curPool), data=self.poolNeighs_[curPool])
            h5File.create_dataset("pool_neighs_hb_"+str(curPool), data=self.poolNeighsHB_[curPool])
            h5File.create_dataset("pool_neighs_start_"+str(curPool), data=self.poolStartNeighs_[curPool])
            h5File.create_dataset("pool_neighs_start_hb_"+str(curPool), data=self.poolStartNeighsHB_[curPool])

        h5File.close()


    def load_pooling_hdf5(self, pFilePath):
        """Method to load a protein pooling from a hdf5 file.

        Args:
            pFilePath (string): File path.
        """
        h5File = h5py.File(pFilePath, "r")
        
        numPoolings = h5File["num_poolings"][()][0]
        self.poolIds_ = []
        self.poolNeighs_ = []
        self.poolNeighsHB_ = []
        self.poolStartNeighs_ = []
        self.poolStartNeighsHB_ = []
        for curPool in range(numPoolings):
            self.poolIds_.append(h5File["pool_ids_"+str(curPool)][()])
            self.poolNeighs_.append(h5File["pool_neighs_"+str(curPool)][()])
            self.poolNeighsHB_.append(h5File["pool_neighs_hb_"+str(curPool)][()])
            self.poolStartNeighs_.append(h5File["pool_neighs_start_"+str(curPool)][()])
            self.poolStartNeighsHB_.append(h5File["pool_neighs_start_hb_"+str(curPool)][()])
            
        h5File.close()

        # def icosahedron():
        #     PHI = (1.0 + np.sqrt(5.0)) / 2.0
        #     sphereLength = np.sqrt(PHI*PHI + 1.0)
        #     dist1 = PHI/sphereLength
        #     dist2 = 1.0/sphereLength

        #     verts = [
        #         [-dist2,  dist1, 0], [ dist2,  dist1, 0], [-dist2, -dist1, 0], [ dist2, -dist1, 0],
        #         [0, -dist2, dist1], [0,  dist2, dist1], [0, -dist2, -dist1], [0,  dist2, -dist1],
        #         [ dist1, 0, -dist2], [ dist1, 0,  dist2], [-dist1, 0, -dist2], [-dist1, 0,  dist2]
        #     ]

        #     faces = [
        #         [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        #         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        #         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        #         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        #     ]

        #     return verts, faces

        
        # sphPts, sphFaces = icosahedron()
        # print(len(self.poolNeighsHB_[0]))
        # print(len(self.covBondListHB_))
        # with open("auxObjFiles/"+pFilePath.split("/")[-1]+".ply", 'w') as myFile:
        #     numvertex = len(sphPts)*len(self.atomPos_[0]) + len(self.covBondListHB_)*3
        #     numfaces = len(sphFaces)*len(self.atomPos_[0]) + len(self.covBondListHB_)*2
        #     myFile.write("ply\n")
        #     myFile.write("format ascii 1.0\n")
        #     myFile.write("element vertex "+ str(numvertex)+"\n")
        #     myFile.write("property float x\n")
        #     myFile.write("property float y\n")
        #     myFile.write("property float z\n")
        #     myFile.write("property uchar red\n")
        #     myFile.write("property uchar green\n")
        #     myFile.write("property uchar blue\n")
        #     myFile.write("element face "+ str(numfaces)+"\n")
        #     myFile.write("property list uchar int vertex_index\n")
        #     myFile.write("end_header\n")

        #     colorList = np.array([
        #         [255, 0, 0],
        #         [255, 125, 0],
        #         [255, 255, 0],
        #         [125, 255, 0],
        #         [0, 255, 0],
        #         [0, 255, 125],
        #         [0, 255, 255],
        #         [0, 125, 255],
        #         [0, 0, 255]])
        #     np.random.seed(0)

        #     for curPosIter, curPos in enumerate(self.atomPos_[0]):
        #         for currSphPt in sphPts:
        #             currPtFlt = [0.5*currSphPt[0]+curPos[0], 0.5*currSphPt[1]+curPos[1], 0.5*currSphPt[2]+curPos[2]]
        #             color = colorList[self.poolIds_[0][curPosIter]%9]
        #             myFile.write(str(currPtFlt[0])+" "+str(currPtFlt[1])+" "+str(currPtFlt[2])+" "+str(color[0])+" "+ str(color[1])+ " "+str(color[2])+"\n")

        #     for curPosIter, curNeighs in enumerate(self.covBondListHB_):
        #         pos1 = self.atomPos_[0][curNeighs[0]]
        #         pos2 = self.atomPos_[0][curNeighs[1]]
        #         vect = np.random.randn(3)
        #         normVect = 0.2/np.linalg.norm(vect)
        #         pos11 = pos1 + vect*normVect
        #         pos12 = pos1 - vect*normVect
        #         myFile.write(str(pos11[0])+" "+str(pos11[1])+" "+str(pos11[2])+" 125 125 125\n")
        #         myFile.write(str(pos12[0])+" "+str(pos12[1])+" "+str(pos12[2])+" 125 125 125\n")
        #         myFile.write(str(pos2[0])+" "+str(pos2[1])+" "+str(pos2[2])+" 125 125 125\n")

        #     offset = 0
        #     for i in range(len(self.atomPos_[0])):
        #         for currSphFace in sphFaces:
        #             myFile.write("3 "+str(currSphFace[0]+offset)+" "+str(currSphFace[1]+offset)+" "+str(currSphFace[2]+offset)+"\n")
        #         offset += len(sphPts)

        #     for curPosIter, curNeighs in enumerate(self.covBondListHB_):
        #         myFile.write("3 "+str(offset)+" "+str(offset+1)+" "+str(offset+2)+"\n")
        #         myFile.write("3 "+str(offset+1)+" "+str(offset)+" "+str(offset+2)+"\n")
        #         offset += 3

        # print(self.covBondListHB_)
        # print(self.poolNeighsHB_[0])
        # with open("auxObjFiles/"+pFilePath.split("/")[-1]+"_simplyfied.ply", 'w') as myFile:

        #     numNodes = len(self.poolStartNeighs_[0])
        #     newCoords = np.full((numNodes, 3), 0.0, dtype= np.float32)
        #     newCoordsCounter = np.full((numNodes), 1.0, dtype= np.float32)
        #     for curIter, curPoolId in enumerate(self.poolIds_[0]):
        #         newCoords[curPoolId, :] += (self.atomPos_[0][curIter] - newCoords[curPoolId, :])/newCoordsCounter[curPoolId]
        #         newCoordsCounter[curPoolId] += 1

        #     numvertex = len(sphPts)*numNodes + len(self.poolNeighsHB_[0])*3
        #     numfaces = len(sphFaces)*numNodes + len(self.poolNeighsHB_[0])*2
        #     myFile.write("ply\n")
        #     myFile.write("format ascii 1.0\n")
        #     myFile.write("element vertex "+ str(numvertex)+"\n")
        #     myFile.write("property float x\n")
        #     myFile.write("property float y\n")
        #     myFile.write("property float z\n")
        #     myFile.write("property uchar red\n")
        #     myFile.write("property uchar green\n")
        #     myFile.write("property uchar blue\n")
        #     myFile.write("element face "+ str(numfaces)+"\n")
        #     myFile.write("property list uchar int vertex_index\n")
        #     myFile.write("end_header\n")

        #     np.random.seed(0)

        #     for curPosIter, curPos in enumerate(newCoords):
        #         for currSphPt in sphPts:
        #             currPtFlt = [0.5*currSphPt[0]+curPos[0], 0.5*currSphPt[1]+curPos[1], 0.5*currSphPt[2]+curPos[2]]
        #             myFile.write(str(currPtFlt[0])+" "+str(currPtFlt[1])+" "+str(currPtFlt[2])+" 125 125 125\n")

        #     for curPosIter, curNeighs in enumerate(self.poolNeighsHB_[0]):
        #         pos1 = newCoords[curNeighs[0]]
        #         pos2 = newCoords[curNeighs[1]]
        #         vect = np.random.randn(3)
        #         normVect = 0.2/np.linalg.norm(vect)
        #         pos11 = pos1 + vect*normVect
        #         pos12 = pos1 - vect*normVect
        #         myFile.write(str(pos11[0])+" "+str(pos11[1])+" "+str(pos11[2])+" 125 125 125\n")
        #         myFile.write(str(pos12[0])+" "+str(pos12[1])+" "+str(pos12[2])+" 125 125 125\n")
        #         myFile.write(str(pos2[0])+" "+str(pos2[1])+" "+str(pos2[2])+" 125 125 125\n")

        #     offset = 0
        #     for i in range(len(newCoords)):
        #         for currSphFace in sphFaces:
        #             myFile.write("3 "+str(currSphFace[0]+offset)+" "+str(currSphFace[1]+offset)+" "+str(currSphFace[2]+offset)+"\n")
        #         offset += len(sphPts)

        #     for curPosIter, curNeighs in enumerate(self.poolNeighsHB_[0]):
        #         myFile.write("3 "+str(offset)+" "+str(offset+1)+" "+str(offset+2)+"\n")
        #         myFile.write("3 "+str(offset+1)+" "+str(offset)+" "+str(offset+2)+"\n")
        #         offset += 3
