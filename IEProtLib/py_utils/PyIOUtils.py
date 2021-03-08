'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyIOUtils.py

    \brief Helper I/O util functions.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import numpy as np
from collections import defaultdict

def visualize_progress(pVal, pMaxVal, pDescription="", pBarWidth=20, pSameLine = False):
    """Method to visualize the progress of a process in the console.

    Args:
        pVal (int): Current step in the process.
        pMaxVal (int): Maximum numbef of step of the process.
        pDescription (string): String to be displayed at the current step.
        pBarWidth (int): Size of the progress bar displayed.
        pSameLine (bool): Boolean that indicates if the progress bar is 
            printed in the same line.
    """

    progress = int((pVal*pBarWidth) / pMaxVal)
    progressBar = ['='] * (progress) + ['>'] + ['.'] * (pBarWidth - (progress+1))
    progressBar = ''.join(progressBar)
    initBar = "%6d/%6d" % (pVal, pMaxVal)
    prefix = "\r" if pSameLine else ""
    end = "" if pSameLine else "\n"
    print(prefix+initBar + ' [' + progressBar + '] ' + pDescription, end=end)
    sys.stdout.flush()
