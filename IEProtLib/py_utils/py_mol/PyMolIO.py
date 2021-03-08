'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyMolIO.py

    \brief Functions to load protein files.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import numpy as np
from collections import defaultdict


def load_protein_pdb(pDBFilePath, pLoadAnim = True, pLoadHydrogens = False, 
    pLoadH2O = False, pLoadGroups = True, pChainFilter = None):
    """Method to load a protein from a PDB file.

    Args:
        pDBFilePath (string): File path to the pdb.
        pLoadAnim (bool): Boolean that indicates if we load the animation.
    """

    atomPos = []
    atomTypes = []
    atomNames = []
    atomResidueIndexs = [] 
    atomResidueType = [] 
    atomChainName = []

    # Parse the PDB file.
    with open(pDBFilePath, 'r') as pdbFile:
        auxAtomResidueIds = []
        auxAltLoc = ""
        dictInsCodes = {}
        for line in pdbFile:

            # Start of a new model.
            if line.startswith("MODEL"):
                atomPos.append([])

            if line.startswith("ENDMDL"):
                # If we only need the first frame.
                if not(pLoadAnim):
                    break
                # If the number of atoms does not match between key frames.
                elif len(atomPos) > 1 and len(atomPos[-1]) != len(atomPos[-2]):
                    atomPos = atomPos[:-1]
                    break

            # Process an atom
            if line.startswith("ATOM") or (line.startswith("HETATM") and pLoadGroups):
                curChainName = line[21:22].strip()
                if pChainFilter is None or curChainName == pChainFilter:

                    curAtomLabel = line[76:78].strip()

                    # If the atom type is not provided, use the first letter of the atom name.
                    # This could lead to errors but the PDB should have the atom type in the 
                    # first place.
                    if len(curAtomLabel) == 0:
                        curAtomLabel = line[12:16].strip()[0]

                    if curAtomLabel != 'H' or pLoadHydrogens:
                        curResidueLabel = line[17:20].strip()

                        if curResidueLabel != 'HOH' or pLoadH2O:
                            if len(atomPos) == 0:
                                atomPos.append([])
                            
                            #Check that is not alternate location definition.
                            curAltLoc = line[16:17]
                            validPosAltLoc = (curAltLoc == " ")
                            if not validPosAltLoc:
                                if auxAltLoc == "":
                                    auxAltLoc = curAltLoc
                                    validPosAltLoc = True
                                else:
                                    validPosAltLoc = (curAltLoc == auxAltLoc)

                            #Check for the  insertion code of the Residue.
                            curICode = line[26:27]
                            curResidueIndex = int(line[22:26].strip())
                            if curResidueIndex in dictInsCodes:
                                validPosICode = dictInsCodes[curResidueIndex] == curICode
                            else:
                                dictInsCodes[curResidueIndex] = curICode
                                validPosICode = True

                            if validPosAltLoc and validPosICode:
                                # Save the atom position.
                                atomXCoord = float(line[30:38].strip())
                                atomYCoord = float(line[38:46].strip())
                                atomZCoord = float(line[46:54].strip())
                                atomPos[-1].append([atomXCoord, atomYCoord, atomZCoord])

                                # Save the topology only in the first model.
                                if len(atomPos) == 1:
                                    # Save the Residue types
                                    atomResidueType.append(curResidueLabel)

                                    # Save the Residue index.
                                    atomResidueIndexs.append(curResidueIndex)

                                    # Save the atom type.
                                    atomTypes.append(curAtomLabel)
                                    
                                    # Save the atom name.
                                    atomNames.append(line[12:16].strip())

                                    # Save the chain name.
                                    if len(curChainName) == 0:
                                        curChainName = 'Z'
                                    atomChainName.append(curChainName)

        # If no atom positions are loaded raise exception.
        if len(atomPos) == 0:
            raise Exception('Empty pdb')
        if len(atomPos[0]) == 0:
            raise Exception('Empty pdb')

        # Transform to numpy arrays
        atomPos = np.array(atomPos)
        atomTypes = np.array(atomTypes)
        atomNames = np.array(atomNames)
        atomResidueIndexs = np.array(atomResidueIndexs)
        atomResidueType = np.array(atomResidueType)
        atomChainName = np.array(atomChainName)

        # Center the molecule
        coordMax = np.amax(atomPos[0], axis=(0))
        coordMin = np.amin(atomPos[0], axis=(0))
        center = (coordMax + coordMin)*0.5
        atomPos = atomPos - center.reshape((1, 1, 3))

    return atomPos, atomTypes, atomNames, atomResidueIndexs, atomResidueType, atomChainName, center

def save_protein_pdb(pFilePath, pProtein):
    """Method to save a protein to a PDB file.

    Args:
        pFilePath (string): Path to the file.
        pProtein (MCPyProtein): Protein to save.
    """
    with open(pFilePath, 'w') as protFile:
        aminoCounter = 0
        lastChainName = "A"
        lastAminoId = -1
        for curAtomIter in range(len(pProtein.atomTypes_)):
            curAtomName = pProtein.atomNames_[curAtomIter]
            while len(curAtomName) < 3:
                curAtomName = curAtomName+" "
            aminoIds = pProtein.atomAminoIds_[curAtomIter]
            resName = pProtein.atomResidueNames_[curAtomIter]
            chainName = pProtein.atomChainNames_[curAtomIter]
            if lastAminoId != aminoIds:
                if lastChainName != chainName:
                    aminoCounter +=2
                else:
                    aminoCounter +=1
            lastChainName = chainName
            lastAminoId = aminoIds
            curAtomPosition = pProtein.atomPos_[0, curAtomIter] + pProtein.center_
            xPosText = "{:8.3f}".format(curAtomPosition[0])
            yPosText = "{:8.3f}".format(curAtomPosition[1])
            zPosText = "{:8.3f}".format(curAtomPosition[2])
            occupancy = "  1.00"
            tempFactor = "  1.00"
            atomType = pProtein.periodicTable_.labels_[pProtein.atomTypes_[curAtomIter]].split("/")[0]
            protFile.write(("ATOM  %5d  %s %s %s%4d    %s%s%s%s%s           %s\n")%(curAtomIter+1, 
                curAtomName, resName, chainName, aminoCounter,
                xPosText, yPosText, zPosText, occupancy, tempFactor, atomType))


def save_molecule_pdb(pFilePath, pMolecule):
    """Method to save a molecule to a PDB file.

    Args:
        pFilePath (string): Path to the file.
        pMolecule (MCPyMolecule): Molecule to save.
    """
    with open(pFilePath, 'w') as protFile:
        for curAtomIter in range(len(pMolecule.atomTypes_)):
            if not pMolecule.atomNames_ is None:
                curAtomName = pMolecule.atomNames_[curAtomIter]
                while len(curAtomName) < 3:
                    curAtomName = curAtomName+" "
            else:
                curAtomName = "  X"
            
            aminoType = "XXX"
            chainName = "A"
            
            curAtomPosition = pMolecule.atomPos_[0, curAtomIter] + pMolecule.center_
            xPosText = "{:8.3f}".format(curAtomPosition[0])
            yPosText = "{:8.3f}".format(curAtomPosition[1])
            zPosText = "{:8.3f}".format(curAtomPosition[2])
            occupancy = "  1.00"
            tempFactor = "  1.00"
            aminoCounter = 1
            atomType = pMolecule.periodicTable_.labels_[pMolecule.atomTypes_[curAtomIter]].split("/")[0]
            protFile.write(("ATOM  %5d  %s %s %s%4d    %s%s%s%s%s           %s\n")%(curAtomIter+1, 
                curAtomName, aminoType, chainName, aminoCounter,
                xPosText, yPosText, zPosText, occupancy, tempFactor, atomType))


def load_protein_mol2(pFilePath, pLoadHydrogens = False, pLoadH2O = False, 
    pLoadGroups = True, pChainFilter = None):
    """Method to load a protein from a Mol2 file.

    Args:
        pFilePath (string): File path to the pdb.
    """

    with open(pFilePath, 'r') as datasetFile:
        # Read the lines of the file.
        lines = datasetFile.readlines()

        # Get the overall information of the molecule.
        splitInitLine = lines[2].split()
        
        # Iterate over the lines.
        atomPos = []
        atomTypes = []
        atomNames = []
        atomResidueIndexs = [] 
        atomResidueName = []
        residueDict = {}
        residueIndexDict = {}
        atomSection = False
        structureSection = False
        for curLine in lines:

            # Check if it is the start of a new section.
            if curLine.startswith("@<TRIPOS>ATOM"):
                atomSection = True
                structureSection = False
            elif curLine.startswith("@<TRIPOS>SUBSTRUCTURE"):
                atomSection = False
                structureSection = True
            elif curLine.startswith("@<TRIPOS>"):
                atomSection = False
                structureSection = False
            else:

                # If we are in the atom section.
                if atomSection:
                    lineElements = curLine.rstrip().split()
                    curAtomName = lineElements[1]
                    curAtomPos = [float(lineElements[2]), 
                        float(lineElements[3]), 
                        float(lineElements[4])]
                    curAtomType = lineElements[5].split('.')[0].upper()
                    if curAtomType != 'H' or pLoadHydrogens:
                        curResidueName = lineElements[7][0:3].upper()
                        if curResidueName != 'HOH' or pLoadH2O:
                            curResidueIndex = int(lineElements[6])

                            # Check if the Residue is valid (it did not appear before).
                            if not(curResidueIndex in residueIndexDict):
                                curVector = []
                                residueIndexDict[curResidueIndex] = []
                            else:
                                curVector = residueIndexDict[curResidueIndex]

                            if not(curAtomName in curVector) and (len(curVector) == 0 or curVector[-1] != '-1'):
                                if curAtomType != "DU":

                                    # Update the temporal dictionary.
                                    residueIndexDict[curResidueIndex].append(curAtomName)
                                    
                                    # Store the atom.
                                    atomPos.append(curAtomPos)
                                    atomTypes.append(curAtomType)
                                    atomNames.append(curAtomName)
                                    atomResidueIndexs.append(curResidueIndex)
                                    atomResidueName.append(curResidueName)
                            else:
                                # Update the temporal dictionary.
                                residueIndexDict[curResidueIndex].append('-1')

                # If we are in the structure section.
                elif structureSection:
                    lineElements = curLine.rstrip().split()
                    if lineElements[3] == "RESIDUE" or pLoadGroups:
                        residueDict[lineElements[0]] = (lineElements[5], lineElements[6])

        # Prepare the final arrays.
        atomResidueType = [] 
        atomChainName = []
        auxAtomMask = np.full((len(atomTypes)), False, dtype=bool)
        for curAtomIter, curResidueIndex in enumerate(atomResidueIndexs):
            curKey = str(curResidueIndex)
            if curKey in residueDict:
                curResidue = residueDict[curKey]
                if pChainFilter is None or pChainFilter == curResidue[0]:
                    atomResidueType.append(curResidue[1])
                    atomChainName.append(curResidue[0])
                    auxAtomMask[curAtomIter] = True


        # Transform to numpy arrays
        atomPos = np.array(atomPos)
        atomTypes = np.array(atomTypes)
        atomNames = np.array(atomNames)
        atomResidueIndexs = np.array(atomResidueIndexs)
        atomResidueType = np.array(atomResidueType)
        atomChainName = np.array(atomChainName)

        atomPos = atomPos[auxAtomMask]
        atomTypes = atomTypes[auxAtomMask]
        atomNames = atomNames[auxAtomMask]
        atomResidueIndexs = atomResidueIndexs[auxAtomMask]

        # Center the molecule
        coordMax = np.amax(atomPos, axis=(0))
        coordMin = np.amin(atomPos, axis=(0))
        center = (coordMax + coordMin)*0.5
        atomPos = atomPos - center.reshape((1, 3))
        atomPos = atomPos.reshape((1, -1, 3))

    return atomPos, atomTypes, atomNames, atomResidueIndexs, atomResidueType, atomChainName, center