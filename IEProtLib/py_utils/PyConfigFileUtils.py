'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyConfigFileUtils.py

    \brief Util function to process the configuration file.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np

def parse_elem_list(pStr, dtype):
    """Method to parse a list of elements in a string format.

    Args:
        pStr (string): List of float values separated by coma.
    Returns:
        (np.array): Array of floats.
    """
    elemsStrList = pStr.split(',')
    elems = []
    for curElem in elemsStrList:
        elems.append(dtype(curElem))
    return np.asarray(elems)