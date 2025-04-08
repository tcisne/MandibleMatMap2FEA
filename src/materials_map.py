import numpy as np
import SimpleITK as sitk
from scipy.spatial import KDTree


def align_and_map(mesh, image_path, voxel_size=0.25):
    """Align mesh with segmented image data."""
    # Load segmented image
    if image_path.endswith(".tif"):
        img = sitk.ReadImage(image_path)
    else:  # DICOM
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_files)
        img = reader.Execute()

    # Get spatial metadata
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())

    # Convert mesh points to image coordinates
    mesh_coords = mesh.points - origin
    voxel_indices = (mesh_coords / spacing).astype(int)

    # Create KDTree for fast spatial queries
    img_array = sitk.GetArrayFromImage(img).T  # Convert to x,y,z order
    tree = KDTree(np.argwhere(img_array > 0) * spacing + origin)

    return img_array, tree
