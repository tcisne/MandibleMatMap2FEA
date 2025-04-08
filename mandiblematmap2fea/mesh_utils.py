"""
Mesh processing utilities for MandibleMatMap2FEA.

This module provides functions for loading and processing mesh files,
generating volumetric meshes, and handling spatial queries.
"""

import os
import numpy as np
import meshio
from scipy.spatial import KDTree

try:
    import tetgen

    TETGEN_AVAILABLE = True
except ImportError:
    TETGEN_AVAILABLE = False

# Try to import optional CAD libraries
try:
    import gmsh

    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


def load_mesh(mesh_file):
    """
    Load mesh file in various formats.

    Supported formats:
    - STL (.stl): Stereolithography
    - OBJ (.obj): Wavefront OBJ
    - PLY (.ply): Polygon File Format
    - VTK (.vtk): Visualization Toolkit
    - OFF (.off): Object File Format
    - STEP (.step, .stp): ISO 10303 STEP
    - IGES (.iges, .igs): Initial Graphics Exchange Specification

    Parameters
    ----------
    mesh_file : str
        Path to mesh file

    Returns
    -------
    meshio.Mesh
        Loaded mesh

    Raises
    ------
    ValueError
        If file format is not supported
    """
    file_ext = os.path.splitext(mesh_file)[1].lower()

    # Handle CAD formats that need special processing
    if file_ext in [".step", ".stp", ".iges", ".igs"]:
        if not GMSH_AVAILABLE:
            raise ImportError(
                f"gmsh is required for loading {file_ext} files. "
                "Install it with 'pip install gmsh'."
            )
        return load_cad_with_gmsh(mesh_file)

    # Use meshio for standard mesh formats
    try:
        return meshio.read(mesh_file)
    except Exception as e:
        raise ValueError(f"Failed to load mesh file: {e}")


def load_cad_with_gmsh(cad_file):
    """
    Load CAD file (STEP, IGES) using gmsh and convert to meshio mesh.

    Parameters
    ----------
    cad_file : str
        Path to CAD file

    Returns
    -------
    meshio.Mesh
        Surface mesh
    """
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Load CAD file
    gmsh.open(cad_file)

    # Generate surface mesh
    gmsh.model.mesh.generate(2)  # 2D mesh (surface)

    # Get mesh data
    nodes, elements, _ = gmsh.model.mesh.getNodes()
    triangles = []

    # Extract triangular elements
    for element_type in gmsh.model.mesh.getElementTypes():
        if gmsh.model.mesh.getElementProperties(element_type)[0] == "Triangle":
            element_tags, element_nodes = gmsh.model.mesh.getElementsByType(
                element_type
            )
            num_nodes_per_element = len(element_nodes) // len(element_tags)
            triangles = (
                element_nodes.reshape(-1, num_nodes_per_element) - 1
            )  # 0-based indexing

    # Get node coordinates
    node_tags, node_coords = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)

    # Finalize gmsh
    gmsh.finalize()

    # Create meshio mesh
    mesh = meshio.Mesh(points=points, cells=[("triangle", triangles)])

    return mesh


def load_stl_mesh(stl_file):
    """
    Load STL mesh file.

    Parameters
    ----------
    stl_file : str
        Path to STL file

    Returns
    -------
    meshio.Mesh
        Loaded mesh
    """
    return load_mesh(stl_file)


def generate_volumetric_mesh(
    surface_mesh, element_size=0.3, mesh_type="tet", quality_threshold=0.1
):
    """
    Generate volumetric mesh from surface mesh.

    Parameters
    ----------
    surface_mesh : meshio.Mesh
        Surface mesh
    element_size : float, optional
        Target element size
    mesh_type : str, optional
        Mesh type ('tet' for tetrahedral)
    quality_threshold : float, optional
        Minimum element quality (0-1)

    Returns
    -------
    meshio.Mesh
        Volumetric mesh

    Raises
    ------
    ImportError
        If tetgen is not available
    ValueError
        If mesh_type is not supported
    """
    if mesh_type != "tet":
        raise ValueError(
            f"Mesh type '{mesh_type}' not supported. Only 'tet' is currently supported."
        )

    if not TETGEN_AVAILABLE:
        raise ImportError(
            "tetgen is required for volumetric mesh generation. "
            "Install it with 'pip install tetgen'."
        )

    # Extract points and faces from surface mesh
    points = surface_mesh.points
    faces = None
    for cell_block in surface_mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            break

    if faces is None:
        raise ValueError("No triangular faces found in surface mesh.")

    # Create tetgen mesh
    tet = tetgen.TetGen(points, faces)

    # Generate volumetric mesh
    # Set max volume based on element_size
    max_volume = element_size**3 / 6  # Approximate volume of a tetrahedron
    tet.tetrahedralize(
        order=1, mindihedral=quality_threshold * 10, maxvolume=max_volume
    )

    # Create meshio mesh from tetgen output
    vol_mesh = meshio.Mesh(points=tet.points, cells=[("tetra", tet.elements)])

    return vol_mesh


def create_spatial_index(points):
    """
    Create KDTree for spatial queries.

    Parameters
    ----------
    points : numpy.ndarray
        Array of 3D points

    Returns
    -------
    scipy.spatial.KDTree
        KDTree for spatial queries
    """
    return KDTree(points)


def get_element_centroids(mesh):
    """
    Calculate centroids of all tetrahedral elements in the mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        Volumetric mesh

    Returns
    -------
    numpy.ndarray
        Array of element centroids
    """
    centroids = []

    # Find tetrahedral elements
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetra_cells = cell_block.data
            break
    else:
        raise ValueError("No tetrahedral elements found in mesh.")

    # Calculate centroids
    for cell in tetra_cells:
        # Get the four vertices of the tetrahedron
        vertices = mesh.points[cell]
        # Calculate centroid
        centroid = np.mean(vertices, axis=0)
        centroids.append(centroid)

    return np.array(centroids)


def map_points_to_image(points, origin, spacing, direction):
    """
    Map points to image coordinates.

    Parameters
    ----------
    points : numpy.ndarray
        Array of 3D points
    origin : numpy.ndarray
        Origin of the image
    spacing : numpy.ndarray
        Voxel spacing
    direction : numpy.ndarray
        Direction cosine matrix

    Returns
    -------
    numpy.ndarray
        Array of voxel indices
    """
    # Transform points to image coordinates
    transformed_points = np.dot(
        np.linalg.inv(direction), (points - origin[:, np.newaxis]).T
    ).T

    # Convert to voxel indices
    voxel_indices = np.round(transformed_points / spacing).astype(int)

    return voxel_indices


def check_mesh_quality(mesh):
    """
    Check the quality of a tetrahedral mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        Volumetric mesh

    Returns
    -------
    dict
        Dictionary with quality metrics
    """
    # Find tetrahedral elements
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetra_cells = cell_block.data
            break
    else:
        raise ValueError("No tetrahedral elements found in mesh.")

    # Calculate quality metrics
    qualities = []
    volumes = []

    for cell in tetra_cells:
        # Get the four vertices of the tetrahedron
        vertices = mesh.points[cell]

        # Calculate volume
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        v3 = vertices[3] - vertices[0]
        volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        volumes.append(volume)

        # Calculate quality (ratio of volume to sum of squared edge lengths)
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edge = np.linalg.norm(vertices[j] - vertices[i])
                edges.append(edge)

        sum_squared_edges = np.sum(np.array(edges) ** 2)
        if sum_squared_edges > 0:
            # Normalized quality metric (0-1, higher is better)
            quality = 12.0 * (3.0 * volume) ** (2.0 / 3.0) / sum_squared_edges
            qualities.append(quality)
        else:
            qualities.append(0.0)

    qualities = np.array(qualities)
    volumes = np.array(volumes)

    return {
        "min_quality": np.min(qualities),
        "max_quality": np.max(qualities),
        "mean_quality": np.mean(qualities),
        "median_quality": np.median(qualities),
        "min_volume": np.min(volumes),
        "max_volume": np.max(volumes),
        "total_volume": np.sum(volumes),
        "num_elements": len(tetra_cells),
    }
