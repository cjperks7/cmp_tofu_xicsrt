'''

Functions to convert between XICSRT and ToFu coordinate representation

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np

__all__ = [
    '_tofu2xicsrt',
    '_xicsrt2tofu'
    ]



# Function to convert the coordinate basis from ToFu to XICSRT
def _tofu2xicsrt(
    data = None,
    ):
    '''
    In comparing the two codes, it is important to unify the global coordinate
    basis when defining sources off the axis of symmetry of the diagnostic

    Consider a point or vector defined by coordinates np.r_[x,y,z] with basis
    x = np.r_[1,0,0], y = np.r_[0,1,0], and z = np.r_[0,0,1]

    We will assume the real-world SPARC diagnostic geometry as good-truth, so
    the origin is the center of the tokamak and looking top-down the diagnostic
    hall is to the East

    In ToFu:
        x -> East
        y -> North
        z -> Up

    In XICSRT:
        x -> North
        y -> Up
        z -> East

    This amounts to two 90degree rotations of the basis with no flipping, so
    doing this transformation does nothing to change the physical results but
    is more "natural" given the terminology employed in XICSRT

    '''

    # Output
    if np.ndim(data) == 1:
        return np.r_[data[1], data[2], data[0]]
    elif np.ndim(data) == 2:
        return np.vstack(
            (data[:,1],
            data[:,2],
            data[:,0]
            )).T

# Function to convert the coordinate basis from XICSRT to ToFu
def _xicsrt2tofu(
    data = None,
    ):
    if np.ndim(data) == 1:
        return np.r_[data[2], data[0], data[1]]
    elif np.ndim(data) == 2:
        return np.vstack(
            (data[:,2],
            data[:,0],
            data[:,1]
            )).T
