'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyPeriodicTable.py

    \brief Object to represent a periodic table.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np 

class PyPeriodicTable:
    """PeriodicTable.

    Attributes:
        labels_ (list): Lables of the atoms.
        mass_ (list): Mass of the atoms.
        covRadius_ (list): Covalent radius of the atoms.
        vdwRadius_ (list): Van der Waals radius of the atoms.
        aLabels_ (list): Labels of the aminoacids.
        aLetters_ (list): Letter of the aminoacids.
    """

    labels_ = np.array(['H/D','HE','LI','BE','B','C','N','O','F','NE','NA','MG','AL','SI','P','S',
        'CL','AR','K','CA','SC','TI','V','CR','MN','FE','CO','NI','CU','ZN','GA','GE','AS',
        'SE','BR','KR','RB','SR','Y','ZR','NB','MO','TC','RU','RH','PD','AG','CD','IN','SN',
        'SB','TE','I','XE','CS','BA','LA','CE','PR','ND','PM','SM','EU','GD','TB','DY','HO',
        'ER','TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU','HG','TL','PB','BI','PO',
        'AT','RN','FR','RA','AC','TH','PA','U','NP','PU','AM','CM','BK','CF','ES','FM','MD',
        'NO','LR','RF','DB','SG','BH','HS','MT','DS','RG','CP','UUT','UUQ','UUP','UUUH','UUS','UUO'])

    mass_ = np.array([1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999, 18.998403163, 20.1797, 
        22.98976928, 24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078, 
        44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 
        69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 
        95.95, 97.0, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 
        126.90447, 131.293, 132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145.0, 
        150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 
        178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 
        207.2, 208.9804, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588, 238.02891, 
        237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 267.0, 270.0, 
        271.0, 270.0, 277.0, 276.0, 281.0, 282.0, 285.0, 285.0, 289.0, 288.0, 293.0, 294.0, 294.0])

    covRadius_ = np.array([0.32, 0.28, 1.34, 0.96, 0.84, 0.72, 0.68, 0.68, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 
        1.036, 1.02, 1.02, 1.06, 2.03, 0.992, 1.70, 1.60, 1.53, 1.39, 1.19, 1.42, 1.11, 1.24, 1.32, 
        1.448, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 
        1.42, 1.39, 1.45, 1.668, 1.42, 1.39, 1.39, 1.38, 1.4, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 
        1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 
        1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 
        1.90, 1.87, 1.80, 1.69, 1.66, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 
        1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57])

    vdwRadius_ = np.array([1.2, 1.4, 1.82, 2.0, 2.0, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 2.0, 2.1, 1.8, 1.8, 
        1.75, 1.88, 2.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.4, 1.39, 1.87, 2.0, 1.85, 1.9, 
        1.85, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.72, 1.58, 1.93, 2.17, 2.0, 2.06, 
        1.98, 2.16, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.75, 1.66, 1.55, 1.96, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
        2.0, 1.86, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    aLabels_ = np.array([
        'HIS/HID/HIE/HIP', #0
        'ASP/ASH', #1
        'ARG/ARN', #2
        'PHE', #3
        'ALA', #4
        'CYS/CYX', #5
        'GLY', #6
        'GLN', #7
        'GLU/GLH', #8
        'LYS/LYN', #9
        'LEU', #10
        'MET', #11
        'ASN', #12
        'SER', #13
        'TYR', #14
        'THR', #15
        'ILE', #16
        'TRP', #17
        'PRO', #18
        'VAL', #19
        'SEC', #20
        'PYL', #21
        'ASX', #22
        'XLE', #23
        'GLX', #24
        'XXX']) #25
    
    aLetters_ = np.array(['H','D','R','F','A','C','G','Q','E','K','L','M','N','S','Y','T','I','W','P','V','U','O','B','J','Z','X'])

    nucleotidesLabels_ = np.array(['DA', 'DC', 'DG', 'DT', 'DI', 'A', 'C', 'G', 'U', 'I'])

    def __init__(self):
        """Constructor.
        """
        pass

    
    def get_num_atoms(self):
        """Method to get the number of atoms.

        Returns:
            int: Number of atoms.
        """
        return len(self.labels_)


    def get_atom_index(self, pAtomType):
        """Method to get the index of the atom based on its type.

        Args:
            pAtomType (string): Atom label.

        Returns:
            int: Index in the list of the atom type.
        """

        index = -1
        for it, atom in enumerate(self.labels_):
            currALabelSplit = atom.split("/")
            for auxCurrLabel in currALabelSplit:
                if pAtomType == auxCurrLabel:
                    index = it

        return index


    def get_atom_label(self, pAtomIndex):
        """Method to get the atom label at position atomIndex.

        Args:
            pAtomIndex (int): Index of the atom in the list.

        Returns:
            string: Atom label.
        """
        if (pAtomIndex > len(self.labels_)) or (pAtomIndex < 0):
            raise RuntimeError('Invalid atom index')

        return self.labels_[pAtomIndex]


    def get_atom_mass(self, pAtomIndex):
        """Method to get the atom mass at position pAtomIndex.

        Args:
            pAtomIndex (int): Index of the atom in the list.

        Returns:
            float: Atom mass.
        """
        if (pAtomIndex > len(self.labels_)) or (pAtomIndex < 0):
            raise RuntimeError('Invalid atom index')

        return self.mass_[pAtomIndex]


    def get_atom_covalent_radius(self, pAtomIndex):
        """Method to get the atom convalent radius at position pAtomIndex.

        Args:
            pAtomIndex (int): Index of the atom in the list.

        Returns:
            float: Atom convalent radius.
        """
        if (pAtomIndex > len(self.labels_)) or (pAtomIndex < 0):
            raise RuntimeError('Invalid atom index')

        return self.covRadius_[pAtomIndex]


    def get_atom_vdw_radius(self, pAtomIndex):
        """Method to get the atom vdw radius at position atomIndex.

        Args:
            pAtomIndex (int): Index of the atom in the list.

        Returns:
            float: Atom vdw radius.
        """
        if (pAtomIndex > len(self.labels_)) or (pAtomIndex < 0):
            raise RuntimeError('Invalid atom index')

        return self.vdwRadius_[pAtomIndex]


    def get_num_aminoacids(self):
        """Method to get the number of aminoacids.

        Returns:
            int: Number of aminoacids.
        """
        return len(self.aLabels_)

    def get_aminoacid_index(self, pALabel):
        """Method to get the index of the aminoacid based on its label.

        Args:
            pALabel (string): Aminoacid label.

        Returns:
            int: Index in the list of the Aminoacid type.
        """

        index = -1
        for it, currALabel in enumerate(self.aLabels_):
            currALabelSplit = currALabel.split("/")
            for auxCurrLabel in currALabelSplit:
                if pALabel == auxCurrLabel:
                    index = it
        return index


    def get_aminoacid_label(self, pAIndex):
        """Method to get the aminoacid label at position aIndex.

        Args:
            pAIndex (int): Index of the aminoacid in the list.

        Returns:
            string: Aminoacid label.
        """
        if (pAIndex > len(self.aLabels_)) or (pAIndex < 0):
            raise RuntimeError('Invalid aminoacid index')

        return self.aLabels_[pAIndex]


    def get_aminoacid_letter(self, pAIndex):
        """Method to get the aminoacid letter at position pAIndex.

        Args:
            aIndex (int): Index of the aminoacid in the list.

        Returns:
            string: Aminoacid letter.
        """
        if (pAIndex > len(self.aLabels_)) or (pAIndex < 0):
            raise RuntimeError('Invalid aminoacid index')

        return self.aLetters_[pAIndex]


    def get_nucleotide_index(self, pNLabel):
        """Method to get the index of the nucleotide based on its label.

        Args:
            pNLabel (string): Nucleotide label.

        Returns:
            int: Index in the list of the Nucleotide type.
        """

        index = -1
        for it, currNLabel in enumerate(self.nucleotidesLabels_):
            if pNLabel == currNLabel:
                index = it
        return index