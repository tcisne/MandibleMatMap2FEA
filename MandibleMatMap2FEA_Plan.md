# MandibleMatMap2FEA: Core Plan

## Overview

MandibleMatMap2FEA is a pipeline for converting segmented CT data of mandibular angle fractures to finite element models with heterogeneous material properties based on Hounsfield Units.

## Core Components

```mermaid
graph LR
    A[Load Input Data] --> B[Process Mesh]
    B --> C[Map Materials]
    C --> D[Export to FEA]
```

### 1. Input Data Processing
- Load DICOM files with Hounsfield Unit data
- Load segmented mandible STL mesh

### 2. Mesh Processing
- Generate volumetric tetrahedral mesh from surface STL
- Prepare mesh for material property mapping

### 3. Material Mapping
- Three-point calibration system (air, dentin, enamel)
- Linear HU to density conversion
- Power law for density to elastic moduli (E = a * density^b)
- Assign properties to mesh elements

### 4. FEA Export
- Generate .inp file with nodes, elements, and material properties
- Group similar materials to reduce file size

## Project Structure

```
MandibleMatMap2FEA/
├── mandiblematmap2fea/
│   ├── __main__.py           # Command-line entry point
│   ├── dicom_utils.py        # DICOM processing functions
│   ├── mesh_utils.py         # Mesh processing functions
│   ├── material_mapping.py   # Material mapping functions
│   ├── fea_export.py         # FEA export functions
│   └── visualization.py      # Basic visualization functions
├── requirements.txt          # Project dependencies
└── README.md                 # Project overview
```

## Workflow

1. **Load and Process Input Data**
   - Load DICOM series and extract HU values
   - Load STL mesh of segmented mandible
   - Generate volumetric mesh

2. **Map Material Properties**
   - Convert HU values to density using three-point calibration
   - Apply power law to convert density to elastic moduli
   - Assign material properties to mesh elements

3. **Export to FEA Format**
   - Generate node and element definitions
   - Export heterogeneous material properties
   - Create .inp file for Abaqus/ANSYS

## Configuration

```yaml
# Core configuration parameters
input:
  dicom_dir: "input/segmented_dicom"
  mandible_mesh: "input/mandible_sur_mesh.stl"

mesh_processing:
  element_size: 0.3  # mm
  mesh_type: "tet"   # tetrahedral elements

material_mapping:
  calibration:
    air:
      hu: -1000
      density: 0.001  # g/cm³
    dentin:
      hu: 2000
      density: 2.14   # g/cm³
    enamel:
      hu: 3000
      density: 2.8    # g/cm³
  power_law:
    cortical:
      a: 2017.3  # Coefficient
      b: 2.5     # Exponent
    trabecular:
      a: 1157.1  # Coefficient
      b: 1.78    # Exponent

fea_export:
  output_file: "output/mandible_simulation.inp"
  element_type: "C3D4"  # Linear tetrahedral
```

## Command-Line Usage

```bash
# Full pipeline
python -m mandiblematmap2fea pipeline --config config.yaml

# Individual stages
python -m mandiblematmap2fea process-mesh --input input/mandible.stl --output output/volumetric_mesh.vtk
python -m mandiblematmap2fea map-materials --mesh output/volumetric_mesh.vtk --dicom input/dicom_dir --output output/mesh_with_materials.vtk
python -m mandiblematmap2fea export --mesh output/mesh_with_materials.vtk --output output/mandible.inp
```

## Core Dependencies

- numpy, scipy: Numerical operations
- SimpleITK: DICOM processing
- meshio: Mesh I/O operations
- pyvista: Visualization (optional)