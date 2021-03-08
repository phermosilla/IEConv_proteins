'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyMolecule.py

    \brief Molecule object.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import numpy as np
import h5py

from IEProtLib.py_utils.py_mol import load_protein_pdb, load_protein_mol2

class PyMolecule:
    """Class to store a molecule.

    Attributes:
        periodicTable_ (MCPyPeriodicTable): Periodic table.
        atomPos_ (float array  kxnx3): Atom positions.
        atomNames_ (string array n): Atom names.
        atomTypes_ (int array n): Atom types.        
    """

    def __init__(self, pPeriodicTable):
        """Constructor.

        Args:
            pPeriodicTable (MCPyPeriodicTable): Periodic table.
        """
        self.periodicTable_ = pPeriodicTable
        self.center_ = None
        self.atomPos_ = None
        self.atomNames_ = None
        self.atomTypes_ = None
        self.covBondList_ = None
        self.atomCovBondSIndices_ = None


    def get_num_atoms(self):
        """Method to get the number of atoms.

        Return:
            (int): Number of atoms.
        """
        return len(self.atomPos_[0])


    def compute_covalent_bonds(self):
        """Method to compute the covalent bonds.
        """

        accumBond = 0
        self.covBondList_ = []
        self.atomCovBondSIndices_ = np.full((len(self.atomPos_[0])), 0, dtype=np.int32)
        covRadii = self.periodicTable_.covRadius_[self.atomTypes_]
        for atomIter in range(len(self.atomPos_[0])):
            curCovRadius = covRadii[atomIter]
            curPos = self.atomPos_[0, atomIter, :]
            dist = np.linalg.norm(curPos.reshape(1, 3) - self.atomPos_[0], axis = 1) \
                - curCovRadius - covRadii - 0.56
            maskAtoms = dist <= 0.0
            maskAtoms[atomIter] = False
            indices = np.where(maskAtoms)[0].reshape(-1, 1)
            indices2 = np.full(indices.shape, atomIter, dtype=np.int32)
            accumBond += indices.shape[0]
            self.covBondList_.append(np.concatenate((indices, indices2), axis=1))
            self.atomCovBondSIndices_[atomIter] = accumBond
        self.covBondList_ = np.concatenate(self.covBondList_, axis=0)


    def load_molecular_file(self, pFilePath, pLoadAnim = True, pFileType = "pdb",
        pLoadHydrogens = False):
        """Method to set the content of the protein from a pdb file.

        Args:
            pFilePath (string): Path to the file.
            pLoadAnim (bool): Boolean that indicates if the animation is loaded.
            pLoadHydrogens (bool): Boolean that indicates if we load the hydrogen atoms.
        """

        if pFileType == "pdb":
            # Load the pdb file.
            atomPos, atomTypes, atomNames, _, _, _, transCenter = load_protein_pdb(pFilePath, pLoadAnim,
                pLoadHydrogens= pLoadHydrogens, pLoadGroups=True)
        elif pFileType == "mol2":
            # Load mol2 file.
            atomPos, atomTypes, atomNames, _, _, _, transCenter = load_protein_mol2(pFilePath,
                pLoadHydrogens= pLoadHydrogens, pLoadGroups=True)

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
        h5File.create_dataset("atom_types", data=self.atomTypes_)
        h5File.create_dataset("atom_names", data=auxAtomNames)
        # Save covalent bonds.
        h5File.create_dataset("cov_bond_list", data=self.covBondList_)
        h5File.create_dataset("cov_bond_list_sindices", data=self.atomCovBondSIndices_)
        h5File.close()


    def load_hdf5(self, pFilePath):
        """Method to load a protein from a hdf5 file.

        Args:
            pFilePath (string): File path.
        """
        h5File = h5py.File(pFilePath, "r")
        self.center_ = h5File["pos_center"][()]
        # Load atoms.
        self.atomPos_ = h5File["atom_pos"][()]
        self.atomTypes_ = h5File["atom_types"][()]
        auxAtomNames = h5File["atom_names"][()]
        self.atomNames_ = np.array([curName.decode('utf8') for curName in auxAtomNames])
        # Load covalent bonds.
        self.covBondList_ = h5File["cov_bond_list"][()]
        self.atomCovBondSIndices_ = h5File["cov_bond_list_sindices"][()]
        h5File.close()