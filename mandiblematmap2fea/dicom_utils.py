"""
DICOM processing utilities for MandibleMatMap2FEA.

This module provides functions for loading and processing DICOM files,
extracting Hounsfield Unit (HU) data, and handling spatial metadata.
"""

import SimpleITK as sitk
import numpy as np


def load_dicom_series(dicom_dir):
    """
    Load DICOM series from directory.

    Parameters
    ----------
    dicom_dir : str
        Path to directory containing DICOM files

    Returns
    -------
    SimpleITK.Image
        Loaded DICOM image
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    return reader.Execute()


def get_hu_array(dicom_image):
    """
    Convert DICOM image to HU array.

    Parameters
    ----------
    dicom_image : SimpleITK.Image
        DICOM image

    Returns
    -------
    numpy.ndarray
        3D array of Hounsfield Units
    """
    return sitk.GetArrayFromImage(dicom_image)


def get_spatial_metadata(dicom_image):
    """
    Extract spatial metadata from DICOM image.

    Parameters
    ----------
    dicom_image : SimpleITK.Image
        DICOM image

    Returns
    -------
    tuple
        (origin, spacing, direction) as numpy arrays
    """
    origin = np.array(dicom_image.GetOrigin())
    spacing = np.array(dicom_image.GetSpacing())
    direction = np.array(dicom_image.GetDirection()).reshape(3, 3)
    return origin, spacing, direction


def world_to_voxel(point, origin, spacing, direction):
    """
    Convert world coordinates to voxel indices.

    Parameters
    ----------
    point : numpy.ndarray
        3D point in world coordinates
    origin : numpy.ndarray
        Origin of the image
    spacing : numpy.ndarray
        Voxel spacing
    direction : numpy.ndarray
        Direction cosine matrix

    Returns
    -------
    numpy.ndarray
        Voxel indices (i, j, k)
    """
    # Transform point to image coordinates
    transformed_point = np.dot(np.linalg.inv(direction), (point - origin))

    # Convert to voxel indices
    voxel_indices = np.round(transformed_point / spacing).astype(int)

    return voxel_indices


def get_hu_value(hu_array, voxel_indices, default_value=-1000):
    """
    Get HU value at specified voxel indices with bounds checking.

    Parameters
    ----------
    hu_array : numpy.ndarray
        3D array of Hounsfield Units
    voxel_indices : numpy.ndarray
        Voxel indices (i, j, k)
    default_value : int, optional
        Default value to return if indices are out of bounds

    Returns
    -------
    float
        HU value at specified voxel
    """
    i, j, k = voxel_indices

    # Check bounds
    if (
        0 <= i < hu_array.shape[0]
        and 0 <= j < hu_array.shape[1]
        and 0 <= k < hu_array.shape[2]
    ):
        return hu_array[i, j, k]
    else:
        return default_value
