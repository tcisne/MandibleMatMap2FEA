"""
Material mapping utilities for MandibleMatMap2FEA.

This module provides functions for mapping material properties to mesh elements
based on Hounsfield Units (HU) from CT data.
"""

import numpy as np


def calibrate_hu_to_density(hu_value, calibration_points):
    """
    Convert HU value to density using linear calibration.

    Parameters
    ----------
    hu_value : float
        Hounsfield Unit value
    calibration_points : dict
        Dictionary with keys 'air', 'dentin', 'enamel'
        Each key maps to a dict with 'hu' and 'density' values

    Returns
    -------
    float
        Density value in g/cm³
    """
    # Extract calibration values
    air_hu = calibration_points["air"]["hu"]
    air_density = calibration_points["air"]["density"]
    dentin_hu = calibration_points["dentin"]["hu"]
    dentin_density = calibration_points["dentin"]["density"]
    enamel_hu = calibration_points["enamel"]["hu"]
    enamel_density = calibration_points["enamel"]["density"]

    # Linear interpolation for values between air and dentin
    if hu_value <= dentin_hu:
        slope = (dentin_density - air_density) / (dentin_hu - air_hu)
        density = air_density + slope * (hu_value - air_hu)
    # Linear interpolation for values between dentin and enamel
    else:
        slope = (enamel_density - dentin_density) / (enamel_hu - dentin_hu)
        density = dentin_density + slope * (hu_value - dentin_hu)

    return density


def density_to_elastic_modulus(density, material_type, power_law_params):
    """
    Convert density to elastic modulus using power law.

    Parameters
    ----------
    density : float
        Density value in g/cm³
    material_type : str
        Material type ('cortical', 'trabecular', 'air')
    power_law_params : dict
        Dictionary with keys matching material_type
        Each key maps to a dict with 'a' and 'b' values for E = a * (density)^b

    Returns
    -------
    float
        Elastic modulus in MPa
    """
    a = power_law_params[material_type]["a"]
    b = power_law_params[material_type]["b"]
    return a * (density**b)


def determine_material_type(density):
    """
    Determine material type based on density.

    Parameters
    ----------
    density : float
        Density value in g/cm³

    Returns
    -------
    str
        Material type ('cortical', 'trabecular', 'air')
    """
    if density < 0.8:
        return "air"
    elif density < 1.8:
        return "trabecular"
    else:
        return "cortical"


def map_materials_to_mesh(
    mesh, hu_array, voxel_indices, calibration_points, power_law_params
):
    """
    Map material properties to mesh elements.

    Parameters
    ----------
    mesh : meshio.Mesh
        Volumetric mesh
    hu_array : numpy.ndarray
        3D array of Hounsfield Units
    voxel_indices : numpy.ndarray
        Array of voxel indices for each element centroid
    calibration_points : dict
        Dictionary with calibration points for HU to density conversion
    power_law_params : dict
        Dictionary with power law parameters for density to elastic modulus conversion

    Returns
    -------
    tuple
        (densities, elastic_moduli) as numpy arrays
    """
    densities = []
    elastic_moduli = []

    # Find tetrahedral elements
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetra_cells = cell_block.data
            break
    else:
        raise ValueError("No tetrahedral elements found in mesh.")

    # Map materials to elements
    for i, voxel_idx in enumerate(voxel_indices):
        # Get HU value (with bounds checking)
        if (
            0 <= voxel_idx[0] < hu_array.shape[0]
            and 0 <= voxel_idx[1] < hu_array.shape[1]
            and 0 <= voxel_idx[2] < hu_array.shape[2]
        ):
            hu_value = hu_array[voxel_idx[0], voxel_idx[1], voxel_idx[2]]
        else:
            hu_value = calibration_points["air"]["hu"]  # Default to air

        # Convert HU to density
        density = calibrate_hu_to_density(hu_value, calibration_points)
        densities.append(density)

        # Determine material type based on density
        material_type = determine_material_type(density)

        # Convert density to elastic modulus
        e_modulus = density_to_elastic_modulus(density, material_type, power_law_params)
        elastic_moduli.append(e_modulus)

    return np.array(densities), np.array(elastic_moduli)


def create_material_groups(elastic_moduli, tolerance=0.05):
    """
    Group elements by similar elastic moduli to reduce file size.

    Parameters
    ----------
    elastic_moduli : numpy.ndarray
        Array of elastic moduli for each element
    tolerance : float, optional
        Relative tolerance for grouping

    Returns
    -------
    dict
        Dictionary mapping representative elastic modulus to list of element indices
    """
    groups = {}
    for i, e_value in enumerate(elastic_moduli):
        # Find closest existing group
        found = False
        for group_e in list(groups.keys()):
            if abs(group_e - e_value) / group_e < tolerance:
                groups[group_e].append(i)
                found = True
                break

        if not found:
            groups[e_value] = [i]

    return groups
