# MandibleMatMap2FEA

A pipeline for converting segmented CT data of mandibular angle fractures to finite element models with heterogeneous material properties based on Hounsfield Units.

## Features

- Load and process medical image data:
  - DICOM files with Hounsfield Unit data
  - TIFF stacks with optional grey value to HU conversion
- Support for multiple mesh/CAD file formats:
  - STL (Stereolithography)
  - OBJ (Wavefront)
  - PLY (Polygon File Format)
  - STEP/STP (ISO 10303 STEP)
  - IGES/IGS (Initial Graphics Exchange Specification)
  - And other formats supported by meshio
- Generate volumetric tetrahedral mesh from surface meshes
- Three-point calibration system (air, dentin, enamel) for HU to density conversion
- Power law relationships for density to elastic moduli conversion
- Heterogeneous material property assignment to mesh elements
- Export to Abaqus INP format for finite element analysis
- Basic visualization capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MandibleMatMap2FEA.git
cd MandibleMatMap2FEA
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

MandibleMatMap2FEA provides multiple entry points to the pipeline, allowing you to start from different stages depending on your needs.

### Full Pipeline

Process a CT stack through the entire pipeline:
```bash
python -m mandiblematmap2fea pipeline --config config.yaml
```

### Individual Pipeline Stages

MandibleMatMap2FEA allows you to run individual stages of the pipeline:

#### 1. Process Mesh Only
```bash
python -m mandiblematmap2fea process-mesh --input input/mandible.stl --output output/volumetric_mesh.vtk --element-size 0.3 --quality 0.1
```

#### 2. Map Materials to Mesh
For DICOM data:
```bash
python -m mandiblematmap2fea map-materials --mesh output/volumetric_mesh.vtk --image input/dicom_dir --output output/mesh_with_materials.vtk --config material_config.yaml
```

For TIFF stack with grey value to HU conversion:
```bash
python -m mandiblematmap2fea map-materials --mesh output/volumetric_mesh.vtk --image input/tiff_stack --image-type tiff --convert-grey --hu-min -1000 --hu-max 3000 --output output/mesh_with_materials.vtk --config material_config.yaml
```

#### 3. Export to FEA Format
```bash
python -m mandiblematmap2fea export --mesh output/mesh_with_materials.vtk --output output/mandible.inp --element-type C3D4
```

#### 4. Visualize Mesh with Materials
```bash
python -m mandiblematmap2fea visualize --mesh output/mesh_with_materials.vtk --output output/visualization.png --property elastic_modulus --colormap bone
```

## Configuration

The pipeline is configured through a YAML file. Here's an example configuration:

```yaml
input:
  # Image data (DICOM or TIFF)
  dicom_dir: "input/segmented_dicom"  # Path to DICOM directory
  # image_path: "input/tiff_stack"    # Alternative: Path to TIFF stack or directory
  # image_type: "dicom"               # Optional: "dicom" or "tiff" (auto-detected if not specified)
  
  # Grey value to HU conversion (for TIFF files)
  convert_grey: false                 # Whether to convert grey values to HU
  # grey_min: null                    # Minimum grey value (null = auto-detect)
  # grey_max: null                    # Maximum grey value (null = auto-detect)
  hu_min: -1000                       # Minimum HU value (air)
  hu_max: 3000                        # Maximum HU value (dense bone/enamel)
  
  # Mesh data
  mandible_mesh: "input/mandible_sur_mesh.stl"  # Path to surface mesh file (STL, OBJ, STEP, etc.)

mesh_processing:
  element_size: 0.3                   # Target element size in mm
  mesh_type: "tet"                    # Mesh type (currently only "tet" supported)
  quality_threshold: 0.1              # Minimum element quality (0-1)

material_mapping:
  calibration:
    air:
      hu: -1000
      density: 0.001                  # g/cm³
    dentin:
      hu: 2000
      density: 2.14                   # g/cm³
    enamel:
      hu: 3000
      density: 2.8                    # g/cm³
  power_law:
    cortical:
      a: 2017.3                       # Coefficient
      b: 2.5                          # Exponent
    trabecular:
      a: 1157.1                       # Coefficient
      b: 1.78                         # Exponent
    air:
      a: 0.001                        # Coefficient
      b: 1.0                          # Exponent

fea_export:
  output_file: "output/mandible_simulation.inp"
  element_type: "C3D4"                # Linear tetrahedral

visualization:
  enabled: true
  output_file: "output/visualization.png"
  colormap: "bone"
  property: "elastic_modulus"         # Alternative: "density"
```

## Pipeline Steps

1. **Load Input Data**
   - Load medical image data (DICOM or TIFF)
   - Convert grey values to HU if needed (for TIFF files)
   - Load segmented mandible mesh (STL, OBJ, STEP, etc.)

2. **Process Mesh**
   - Generate volumetric tetrahedral mesh from surface mesh
   - Check mesh quality and report metrics
   - Calculate element centroids

3. **Map Materials**
   - Convert HU values to density using three-point calibration
   - Apply power law to convert density to elastic moduli
   - Assign material properties to mesh elements

4. **Export to FEA**
   - Generate node and element definitions
   - Group similar materials to reduce file size
   - Create INP file for Abaqus/ANSYS

## Dependencies

- numpy, scipy: Numerical operations
- SimpleITK: DICOM processing
- tifffile: TIFF stack processing
- meshio: Mesh I/O operations
- tetgen: Volumetric mesh generation
- gmsh (optional): CAD file processing (STEP, IGES)
- pyvista: Visualization
- matplotlib: 2D visualization
- pyyaml: Configuration file parsing

## License

This project is licensed under the MIT License - see the LICENSE file for details.