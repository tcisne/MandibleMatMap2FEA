"""
Image processing utilities for MandibleMatMap2FEA.

This module provides functions for loading and processing medical image data,
including DICOM and TIFF stacks, extracting Hounsfield Unit (HU) data,
and handling spatial metadata.
"""

import os
import glob
import SimpleITK as sitk
import numpy as np
import tifffile


def load_image_data(image_path, image_type=None):
    """
    Load image data from various formats (DICOM series or TIFF stack).

    Parameters
    ----------
    image_path : str
        Path to directory containing DICOM files or TIFF stack
    image_type : str, optional
        Type of image data ('dicom', 'tiff'). If None, it will be inferred from the path.

    Returns
    -------
    SimpleITK.Image
        Loaded image
    """
    # Infer image type if not provided
    if image_type is None:
        if os.path.isdir(image_path):
            # Check if directory contains DICOM files
            dicom_files = glob.glob(os.path.join(image_path, "*.dcm"))
            if dicom_files:
                image_type = "dicom"
            else:
                # Check if directory contains TIFF files
                tiff_files = glob.glob(os.path.join(image_path, "*.tif")) + glob.glob(
                    os.path.join(image_path, "*.tiff")
                )
                if tiff_files:
                    image_type = "tiff"
                else:
                    raise ValueError(
                        f"Could not determine image type from directory: {image_path}"
                    )
        elif image_path.lower().endswith((".tif", ".tiff")):
            image_type = "tiff"
        elif image_path.lower().endswith(".dcm"):
            image_type = "dicom"
        else:
            raise ValueError(f"Could not determine image type from path: {image_path}")

    # Load image based on type
    if image_type.lower() == "dicom":
        return load_dicom_series(image_path)
    elif image_type.lower() == "tiff":
        return load_tiff_stack(image_path)
    else:
        raise ValueError(f"Unsupported image type: {image_type}")


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


def load_tiff_stack(tiff_path):
    """
    Load TIFF stack from file or directory.

    Parameters
    ----------
    tiff_path : str
        Path to TIFF file or directory containing TIFF files

    Returns
    -------
    SimpleITK.Image
        Loaded TIFF image as SimpleITK image
    """
    if os.path.isdir(tiff_path):
        # Load TIFF files from directory
        tiff_files = sorted(
            glob.glob(os.path.join(tiff_path, "*.tif"))
            + glob.glob(os.path.join(tiff_path, "*.tiff"))
        )
        if not tiff_files:
            raise ValueError(f"No TIFF files found in directory: {tiff_path}")

        # Load first image to get dimensions
        first_img = tifffile.imread(tiff_files[0])

        # Create 3D array to hold all slices
        img_array = np.zeros(
            (len(tiff_files), first_img.shape[0], first_img.shape[1]),
            dtype=first_img.dtype,
        )

        # Load all slices
        for i, file in enumerate(tiff_files):
            img_array[i] = tifffile.imread(file)
    else:
        # Load single TIFF file (which might be a stack)
        img_array = tifffile.imread(tiff_path)

        # Ensure 3D array
        if img_array.ndim == 2:
            img_array = img_array[np.newaxis, :, :]

    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(img_array)

    # Set default spacing if not available
    sitk_image.SetSpacing([1.0, 1.0, 1.0])
    sitk_image.SetOrigin([0.0, 0.0, 0.0])

    return sitk_image


def grey_to_hu(grey_array, hu_min=-1000, hu_max=3000, grey_min=None, grey_max=None):
    """
    Convert grey values to Hounsfield Units using linear mapping.

    Parameters
    ----------
    grey_array : numpy.ndarray
        Array of grey values
    hu_min : float, optional
        Minimum HU value (default: -1000, air)
    hu_max : float, optional
        Maximum HU value (default: 3000, dense bone/enamel)
    grey_min : float, optional
        Minimum grey value. If None, uses the minimum value in grey_array.
    grey_max : float, optional
        Maximum grey value. If None, uses the maximum value in grey_array.

    Returns
    -------
    numpy.ndarray
        Array of Hounsfield Units
    """
    # Determine grey value range if not provided
    if grey_min is None:
        grey_min = np.min(grey_array)
    if grey_max is None:
        grey_max = np.max(grey_array)

    # Avoid division by zero
    if grey_max == grey_min:
        return np.ones_like(grey_array) * hu_min

    # Linear mapping from grey values to HU
    hu_array = hu_min + (grey_array - grey_min) * (hu_max - hu_min) / (
        grey_max - grey_min
    )

    return hu_array


def get_hu_array(
    image,
    convert_from_grey=False,
    hu_min=-1000,
    hu_max=3000,
    grey_min=None,
    grey_max=None,
):
    """
    Convert image to HU array.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image
    convert_from_grey : bool, optional
        Whether to convert from grey values to HU
    hu_min : float, optional
        Minimum HU value for conversion
    hu_max : float, optional
        Maximum HU value for conversion
    grey_min : float, optional
        Minimum grey value for conversion
    grey_max : float, optional
        Maximum grey value for conversion

    Returns
    -------
    numpy.ndarray
        3D array of Hounsfield Units
    """
    # Get array from image
    array = sitk.GetArrayFromImage(image)

    # Convert from grey values to HU if requested
    if convert_from_grey:
        array = grey_to_hu(array, hu_min, hu_max, grey_min, grey_max)

    return array


def get_spatial_metadata(image):
    """
    Extract spatial metadata from image.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image

    Returns
    -------
    tuple
        (origin, spacing, direction) as numpy arrays
    """
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
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
