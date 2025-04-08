from mesh_tools import generate_volumetric_mesh
from material_mapper import align_and_map
from inp_writer import export_to_inp

def run_pipeline():
    # 1. Generate volumetric mesh
    mesh = generate_volumetric_mesh(
        "input/modified_mandible.stl",
        element_size=0.3,
        mesh_type="tet"
    )
    
    # 2. Align with segmented image data
    img_array, kdtree = align_and_map(
        mesh,
        "input/segmented_dicom",
        voxel_size=0.25
    )
    
    # 3. Map material properties
    densities = []
    for i, cell in enumerate(mesh.cells_dict["tetra"]:
        centroid = np.mean(mesh.points[cell], axis=0)
        _, nearest_voxel = kdtree.query(centroid)
        densities.append(img_array[tuple(nearest_voxel.astype(int))])
    
    # 4. Export to INP
    export_to_inp(mesh, densities, "output/mandible_simulation.inp")
