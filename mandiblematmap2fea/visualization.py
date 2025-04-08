"""
Visualization utilities for MandibleMatMap2FEA.

This module provides functions for visualizing mesh and material properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def visualize_material_properties(
    mesh,
    property_values,
    output_file=None,
    property_name="Elastic Modulus",
    colormap="bone",
    log_scale=True,
):
    """
    Visualize material properties on mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        Volumetric mesh
    property_values : numpy.ndarray
        Array of property values for each element
    output_file : str, optional
        Path to output image file
    property_name : str, optional
        Name of the property being visualized
    colormap : str, optional
        Matplotlib colormap name
    log_scale : bool, optional
        Whether to use logarithmic scale for colormap

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If pyvista is not available
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError(
            "pyvista is required for visualization. "
            "Install it with 'pip install pyvista'."
        )

    # Convert meshio mesh to pyvista mesh
    points = mesh.points

    # Find tetrahedral elements
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetra_cells = cell_block.data
            break
    else:
        raise ValueError("No tetrahedral elements found in mesh.")

    # Create connectivity array for pyvista
    # Format: [4, i0, i1, i2, i3, 4, j0, j1, j2, j3, ...]
    n_cells = len(tetra_cells)
    conn = np.zeros(5 * n_cells, dtype=int)
    conn[0::5] = 4  # Number of points per cell
    conn[1::5] = tetra_cells[:, 0]
    conn[2::5] = tetra_cells[:, 1]
    conn[3::5] = tetra_cells[:, 2]
    conn[4::5] = tetra_cells[:, 3]

    # Create pyvista mesh
    pv_mesh = pv.UnstructuredGrid(conn, np.array([10] * n_cells, dtype=int), points)

    # Add property values as cell data
    pv_mesh.cell_data[property_name] = property_values

    # Create plotter
    plotter = pv.Plotter(off_screen=output_file is not None)

    # Set up colormap
    if log_scale and np.min(property_values) > 0:
        # Logarithmic scale
        norm = mcolors.LogNorm(
            vmin=np.min(property_values), vmax=np.max(property_values)
        )
        pv_mesh.cell_data[property_name] = np.log10(property_values)
        scalar_bar_title = f"log10({property_name})"
    else:
        # Linear scale
        norm = mcolors.Normalize(
            vmin=np.min(property_values), vmax=np.max(property_values)
        )
        scalar_bar_title = property_name

    # Add mesh to plotter
    plotter.add_mesh(
        pv_mesh,
        scalars=property_name,
        cmap=colormap,
        show_edges=False,
        scalar_bar_args={"title": scalar_bar_title},
    )

    # Set background and camera position
    plotter.set_background("white")
    plotter.view_isometric()

    # Save or show
    if output_file:
        plotter.screenshot(output_file, transparent_background=True)
        plotter.close()
    else:
        plotter.show()


def visualize_slice(
    hu_array,
    slice_idx=None,
    axis=0,
    output_file=None,
    colormap="gray",
    vmin=-1000,
    vmax=3000,
):
    """
    Visualize a slice of the HU array.

    Parameters
    ----------
    hu_array : numpy.ndarray
        3D array of Hounsfield Units
    slice_idx : int, optional
        Index of the slice to visualize. If None, the middle slice is used.
    axis : int, optional
        Axis along which to take the slice (0, 1, or 2)
    output_file : str, optional
        Path to output image file
    colormap : str, optional
        Matplotlib colormap name
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap

    Returns
    -------
    None
    """
    # Get slice
    if slice_idx is None:
        slice_idx = hu_array.shape[axis] // 2

    if axis == 0:
        slice_data = hu_array[slice_idx, :, :]
    elif axis == 1:
        slice_data = hu_array[:, slice_idx, :]
    else:
        slice_data = hu_array[:, :, slice_idx]

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(slice_data, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Hounsfield Units")
    plt.title(f"CT Slice (axis={axis}, index={slice_idx})")

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
