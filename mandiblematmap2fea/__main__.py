"""
Command-line entry point for MandibleMatMap2FEA.

This module provides the command-line interface for the MandibleMatMap2FEA pipeline.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import meshio

from . import image_utils
from . import mesh_utils
from . import material_mapping
from . import fea_export
from . import visualization


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MandibleMatMap2FEA: Convert CT data to FEA models with heterogeneous material properties"
    )

    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Full pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument(
        "--config", required=True, help="Path to configuration file"
    )

    # Individual stages
    mesh_parser = subparsers.add_parser("process-mesh", help="Process mesh only")
    mesh_parser.add_argument(
        "--input", required=True, help="Input mesh file (STL, OBJ, STEP, etc.)"
    )
    mesh_parser.add_argument(
        "--output", required=True, help="Output volumetric mesh file"
    )
    mesh_parser.add_argument(
        "--element-size", type=float, default=0.3, help="Element size"
    )
    mesh_parser.add_argument(
        "--quality", type=float, default=0.1, help="Minimum element quality (0-1)"
    )

    material_parser = subparsers.add_parser("map-materials", help="Map materials only")
    material_parser.add_argument("--mesh", required=True, help="Input mesh file")
    material_parser.add_argument(
        "--image", required=True, help="DICOM directory or TIFF stack"
    )
    material_parser.add_argument(
        "--image-type",
        choices=["dicom", "tiff"],
        help="Image type (auto-detected if not specified)",
    )
    material_parser.add_argument(
        "--output", required=True, help="Output file with materials"
    )
    material_parser.add_argument(
        "--config", required=True, help="Material configuration file"
    )
    material_parser.add_argument(
        "--convert-grey", action="store_true", help="Convert grey values to HU"
    )
    material_parser.add_argument(
        "--grey-min", type=float, help="Minimum grey value for conversion"
    )
    material_parser.add_argument(
        "--grey-max", type=float, help="Maximum grey value for conversion"
    )
    material_parser.add_argument(
        "--hu-min", type=float, default=-1000, help="Minimum HU value for conversion"
    )
    material_parser.add_argument(
        "--hu-max", type=float, default=3000, help="Maximum HU value for conversion"
    )

    export_parser = subparsers.add_parser("export", help="Export to FEA format")
    export_parser.add_argument(
        "--mesh", required=True, help="Input mesh file with materials"
    )
    export_parser.add_argument("--output", required=True, help="Output INP file")
    export_parser.add_argument("--element-type", default="C3D4", help="Element type")

    vis_parser = subparsers.add_parser(
        "visualize", help="Visualize mesh with materials"
    )
    vis_parser.add_argument(
        "--mesh", required=True, help="Input mesh file with materials"
    )
    vis_parser.add_argument("--output", required=True, help="Output image file")
    vis_parser.add_argument(
        "--property", default="elastic_modulus", help="Property to visualize"
    )
    vis_parser.add_argument("--colormap", default="bone", help="Colormap to use")

    return parser.parse_args()


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def ensure_directory_exists(file_path):
    """Ensure the directory for the given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def run_pipeline(config_file):
    """Run the full pipeline."""
    print(f"Loading configuration from {config_file}")
    config = load_config(config_file)

    # Create output directory if it doesn't exist
    output_file = config["fea_export"]["output_file"]
    ensure_directory_exists(output_file)

    # 1. Load image data (DICOM or TIFF)
    image_path = config["input"].get("dicom_dir") or config["input"].get("image_path")
    image_type = config["input"].get("image_type")
    convert_grey = config["input"].get("convert_grey", False)

    print(f"Loading image data from {image_path}")
    image = image_utils.load_image_data(image_path, image_type)

    # Get HU array with optional grey value conversion
    if convert_grey:
        print("Converting grey values to HU")
        grey_min = config["input"].get("grey_min")
        grey_max = config["input"].get("grey_max")
        hu_min = config["input"].get("hu_min", -1000)
        hu_max = config["input"].get("hu_max", 3000)

        hu_array = image_utils.get_hu_array(
            image,
            convert_from_grey=True,
            hu_min=hu_min,
            hu_max=hu_max,
            grey_min=grey_min,
            grey_max=grey_max,
        )
    else:
        hu_array = image_utils.get_hu_array(image)

    origin, spacing, direction = image_utils.get_spatial_metadata(image)

    # 2. Load and process mesh
    mesh_path = config["input"]["mandible_mesh"]
    print(f"Loading mesh from {mesh_path}")
    surface_mesh = mesh_utils.load_mesh(mesh_path)

    print("Generating volumetric mesh")
    volumetric_mesh = mesh_utils.generate_volumetric_mesh(
        surface_mesh,
        element_size=config["mesh_processing"]["element_size"],
        mesh_type=config["mesh_processing"]["mesh_type"],
        quality_threshold=config["mesh_processing"].get("quality_threshold", 0.1),
    )

    # Check mesh quality
    print("Checking mesh quality")
    quality_metrics = mesh_utils.check_mesh_quality(volumetric_mesh)
    print(f"  Minimum element quality: {quality_metrics['min_quality']:.4f}")
    print(f"  Mean element quality: {quality_metrics['mean_quality']:.4f}")
    print(f"  Number of elements: {quality_metrics['num_elements']}")
    print(f"  Total volume: {quality_metrics['total_volume']:.4f}")

    # 3. Calculate element centroids
    print("Calculating element centroids")
    centroids = mesh_utils.get_element_centroids(volumetric_mesh)

    # 4. Map centroids to image coordinates
    print("Mapping centroids to image coordinates")
    voxel_indices = np.array(
        [
            image_utils.world_to_voxel(centroid, origin, spacing, direction)
            for centroid in centroids
        ]
    )

    # 5. Map materials to mesh
    print("Mapping materials to mesh")
    densities, elastic_moduli = material_mapping.map_materials_to_mesh(
        volumetric_mesh,
        hu_array,
        voxel_indices,
        config["material_mapping"]["calibration"],
        config["material_mapping"]["power_law"],
    )

    # 6. Export to INP format
    print(f"Exporting to INP format: {output_file}")
    fea_export.write_inp_file(
        volumetric_mesh,
        elastic_moduli,
        output_file,
        element_type=config["fea_export"]["element_type"],
    )

    # 7. Save mesh with material properties for visualization
    vtk_file = os.path.splitext(output_file)[0] + ".vtk"
    print(f"Saving mesh with material properties: {vtk_file}")
    fea_export.write_vtk_file(volumetric_mesh, densities, elastic_moduli, vtk_file)

    # 8. Visualize results if enabled
    if config.get("visualization", {}).get("enabled", False):
        vis_file = os.path.splitext(output_file)[0] + ".png"
        print(f"Generating visualization: {vis_file}")
        visualization.visualize_material_properties(
            volumetric_mesh,
            elastic_moduli,
            vis_file,
            property_name=config["visualization"].get("property", "elastic_modulus"),
            colormap=config["visualization"].get("colormap", "bone"),
        )

    print("Pipeline completed successfully")


def process_mesh(args):
    """Process mesh only."""
    print(f"Loading mesh from {args.input}")
    surface_mesh = mesh_utils.load_mesh(args.input)

    print("Generating volumetric mesh")
    volumetric_mesh = mesh_utils.generate_volumetric_mesh(
        surface_mesh, element_size=args.element_size, quality_threshold=args.quality
    )

    # Check mesh quality
    print("Checking mesh quality")
    quality_metrics = mesh_utils.check_mesh_quality(volumetric_mesh)
    print(f"  Minimum element quality: {quality_metrics['min_quality']:.4f}")
    print(f"  Mean element quality: {quality_metrics['mean_quality']:.4f}")
    print(f"  Number of elements: {quality_metrics['num_elements']}")
    print(f"  Total volume: {quality_metrics['total_volume']:.4f}")

    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output)

    # Save volumetric mesh
    print(f"Saving volumetric mesh to {args.output}")
    meshio.write(args.output, volumetric_mesh)

    print("Mesh processing completed successfully")


def map_materials(args):
    """Map materials to mesh."""
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load mesh
    print(f"Loading mesh from {args.mesh}")
    mesh = meshio.read(args.mesh)

    # Load image data (DICOM or TIFF)
    print(f"Loading image data from {args.image}")
    image = image_utils.load_image_data(args.image, args.image_type)

    # Get HU array with optional grey value conversion
    if args.convert_grey:
        print("Converting grey values to HU")
        hu_array = image_utils.get_hu_array(
            image,
            convert_from_grey=True,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            grey_min=args.grey_min,
            grey_max=args.grey_max,
        )
    else:
        hu_array = image_utils.get_hu_array(image)

    origin, spacing, direction = image_utils.get_spatial_metadata(image)

    # Calculate element centroids
    print("Calculating element centroids")
    centroids = mesh_utils.get_element_centroids(mesh)

    # Map centroids to image coordinates
    print("Mapping centroids to image coordinates")
    voxel_indices = np.array(
        [
            image_utils.world_to_voxel(centroid, origin, spacing, direction)
            for centroid in centroids
        ]
    )

    # Map materials to mesh
    print("Mapping materials to mesh")
    densities, elastic_moduli = material_mapping.map_materials_to_mesh(
        mesh,
        hu_array,
        voxel_indices,
        config["material_mapping"]["calibration"],
        config["material_mapping"]["power_law"],
    )

    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output)

    # Save mesh with material properties
    print(f"Saving mesh with material properties to {args.output}")
    fea_export.write_vtk_file(mesh, densities, elastic_moduli, args.output)

    print("Material mapping completed successfully")


def export_to_fea(args):
    """Export mesh with materials to FEA format."""
    # Load mesh with material properties
    print(f"Loading mesh from {args.mesh}")
    mesh = meshio.read(args.mesh)

    # Extract elastic moduli from cell data
    if "elastic_modulus" not in mesh.cell_data:
        print("Error: Mesh does not contain elastic modulus data")
        sys.exit(1)

    elastic_moduli = mesh.cell_data["elastic_modulus"][0]

    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output)

    # Export to INP format
    print(f"Exporting to INP format: {args.output}")
    fea_export.write_inp_file(
        mesh, elastic_moduli, args.output, element_type=args.element_type
    )

    print("FEA export completed successfully")


def visualize_mesh(args):
    """Visualize mesh with materials."""
    # Load mesh with material properties
    print(f"Loading mesh from {args.mesh}")
    mesh = meshio.read(args.mesh)

    # Extract property values from cell data
    if args.property not in mesh.cell_data:
        print(f"Error: Mesh does not contain {args.property} data")
        sys.exit(1)

    property_values = mesh.cell_data[args.property][0]

    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output)

    # Visualize
    print(f"Generating visualization: {args.output}")
    visualization.visualize_material_properties(
        mesh,
        property_values,
        args.output,
        property_name=args.property,
        colormap=args.colormap,
    )

    print("Visualization completed successfully")


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "pipeline":
        run_pipeline(args.config)
    elif args.command == "process-mesh":
        process_mesh(args)
    elif args.command == "map-materials":
        map_materials(args)
    elif args.command == "export":
        export_to_fea(args)
    elif args.command == "visualize":
        visualize_mesh(args)
    else:
        print("Error: No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
