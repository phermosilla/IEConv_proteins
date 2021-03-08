'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyPCUtils.py

    \brief Util functions to process point clouds.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np

def jitter_pc(pRandomState, pPts, pNoise=0.001, pClip=0.005):
    """Method to jitter a point cloud.

    Args:
        pRandomState (np.RandomState): Random state.
        pPts (float np.array nxd): Point cloud.
        pNoise (float): Noise added to the point coordinates.
        pClip (float): Value to clip the noise.
    Returns:
        (float np.array nxd): Mirrored points.
    """
    noise = np.clip(pRandomState.randn(pPts.shape[0], pPts.shape[1])*pNoise,
        -1.0*pClip, pClip)
    return pPts + noise


def mirror_pc(pRandomState, pAxis, pPts, pNormals=None):
    """Method to mirror axis of the point cloud.

    Args:
        pRandomState (np.RandomState): Random state.
        pAxis (bool list): Boolean list that indicates which axis can 
            be mirrored.
        pPts (float np.array nxd): Point cloud.
        pNormals (float n.parray nxd): Point normals.
    Returns:
        (float np.array nxd): Mirrored points.
        (float np.array nxd): Mirrored normals.
    """
    mulVals = np.full((1, pPts.shape[1]), 1.0, dtype=np.float32)
    for curAxis  in range(len(pAxis)):
        if pRandomState.random_sample() > 0.5 and pAxis[curAxis]:
            mulVals[0, curAxis] = -1.0
    retPts = pPts*mulVals
    retNormals = None
    if not(pNormals is None):
        retNormals = pNormals*mulVals
    return retPts, retNormals


def anisotropic_sacle_pc(pRandomState, pPts, pMinScale = 0.9, pMaxScale = 1.1, pReturnScaling = False):
    """Method to scale a model with anisotropic scaling.

    Args:
        pRandomState (np.RandomState): Random state.
        pPts (float np.array nxd): Point cloud.
        pMinScale (float): Minimum scaling.
        pMaxScale (float): Maximum scaling.
        pReturnScaling (bool): Boolean that indicates if the scaling is also returned.
    Returns:
        (float tensor nxd): Scaled point cloud.
    """

    #Total scaling range.
    scaleRange = pMaxScale - pMinScale
    curScaling = pRandomState.random_sample((1, pPts.shape[1]))*scaleRange + pMinScale

    #Return the scaled points.
    if pReturnScaling:
        return pPts*curScaling, curScaling
    else:
        return pPts*curScaling


def symmetric_deformation_pc(self, pRandomState, pPts, pNumDivisions = 4, pNoiseLevel = 0.33):
    """Method to deform the model in blocks symmetrically.

    Args:
        pRandomState (np.RandomState): Random state.
        pPts (float np.array nxd): Point cloud.
        pNumDivisions (int): Number of division applied to the 
            bounding box. This number should be even.
        pNoiseLevel (float): Level of noise applied to the models.
    Returns:
        (float tensor nxd): Deformed point cloud.
    """
    # numDivisions should be even number

    #Compute AABB
    coordMin = np.amin(inData,axis=0)
    coordMax = np.amax(inData,axis=0)

    outData = np.empty(pPts.shape)
    
    for j in range(pPts.shape[1]):
        b = np.linspace(0,1,pNumDivisions)*(coordMax[j]-coordMin[j])+coordMin[j]
        s = np.zeros(pNumDivisions)
        s[1:pNumDivisions//2] = (pRandomState.randn(float(pNumDivisions)/2.0-1.0) \
                        *pNoiseLevel)/float(pNumDivisions)
        s[pNumDivisions//2:-1] = -s[1:pNumDivisions//2][::-1]
        np.clip(s,-0.5/float(pNumDivisions), 0.5/float(pNumDivisions), out=s)
        s = s*(coordMax[j]-coordMin[j])
        bn = b + s

        for i in range(1, pNumDivisions):
            aff_v = b[i-1]<=pPts[:,j]
            aff_v *= pPts[:,j] <= b[i]
            m = (bn[i-1]-bn[i])/(b[i-1]-b[i])
            outData[aff_v,j] = pPts[aff_v,j]*m + bn[i-1] - b[i-1] * m
            
    return outData


def rotate_pc_3d(pRandomState, pPts, pMaxAngle = 2.0 * np.pi, pAxes = [0, 1, 2]):
    """Method to rotate a 3D point cloud.

    Args:
        pRandomState (np.RandomState): Random state.
        pPts (float np.array nx3): Point cloud.
        pMaxAngle (float): Max rotation angle.
        pAxes (list of ints): Axis for which we compute the rotation 
            (0:X, 1:Y, 2:Z).
    Returns:
        (float np.array nx3): Rotated point cloud.
        (float np.array nx3): Rotated point cloud normals.
        (float np.array 3x3): Rotation matrix.
    """

    #Compute the rotation matrix.
    angles = (pRandomState.random_sample((3))-0.5) * 2.0 * pMaxAngle
    if 0 in pAxes:
        Rx = np.array([[1.0, 0.0, 0.0],
                [0.0, np.cos(angles[0]), -np.sin(angles[0])],
                [0.0, np.sin(angles[0]), np.cos(angles[0])]])
    else:
        Rx = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    if 1 in pAxes:
        Ry = np.array([[np.cos(angles[1]), 0.0, np.sin(angles[1])],
                [0.0, 1.0, 0.0],
                [-np.sin(angles[1]), 0.0, np.cos(angles[1])]])
    else:
        Ry = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    if 2 in pAxes:
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0.0],
                [np.sin(angles[2]), np.cos(angles[2]), 0.0],
                [0.0, 0.0, 1.0]])
    else:
        Rz = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    rotationMatrix = np.dot(Rz, np.dot(Ry,Rx))

    #Compute the rotated point cloud.
    return np.dot(pPts, rotationMatrix), rotationMatrix


def rotate_pc_3d_with_normals(pRandomState, pPts, pNormals, pMaxAngle = 2.0 * np.pi, pAxes = [0, 1, 2]):
    """Method to rotate a 3D point cloud.

    Args:
        pRandomState (np.RandomState): Random state.
        pPts (float np.array nx3): Point cloud.
        pNormals (float np.array nx3): Point cloud normals.
        pMaxAngle (float): Max rotation angle.
        pAxes (list of ints): Axis for which we compute the rotation 
            (0:X, 1:Y, 2:Z).
    Returns:
        (float np.array nx3): Rotated point cloud.
        (float np.array nx3): Rotated point cloud normals.
        (float np.array 3x3): Rotation matrix.
    """

    rotPts, rotMat = rotate_pc_3d(pRandomState, pPts, pMaxAngle, pAxes)

    #Compute the rotated point cloud.
    return rotPts, np.dot(pNormals, rotMat), rotMat
