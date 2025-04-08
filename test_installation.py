"""
Test script to verify that the MandibleMatMap2FEA package is installed correctly.
"""

import sys
import importlib.util


def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def main():
    """Main function to test the installation."""
    print("Testing MandibleMatMap2FEA installation...")

    # Check core modules
    core_modules = [
        "mandiblematmap2fea",
        "mandiblematmap2fea.dicom_utils",
        "mandiblematmap2fea.mesh_utils",
        "mandiblematmap2fea.material_mapping",
        "mandiblematmap2fea.fea_export",
        "mandiblematmap2fea.visualization",
    ]

    all_passed = True
    for module in core_modules:
        if check_module(module):
            print(f"✓ {module} imported successfully")
        else:
            print(f"✗ {module} import failed")
            all_passed = False

    # Check dependencies
    dependencies = [
        "numpy",
        "scipy",
        "SimpleITK",
        "meshio",
        "pyvista",
        "matplotlib",
        "yaml",
    ]

    print("\nChecking dependencies:")
    for dep in dependencies:
        if check_module(dep):
            print(f"✓ {dep} imported successfully")
        else:
            print(f"✗ {dep} import failed")
            all_passed = False

    # Optional dependencies
    optional_deps = ["tetgen"]

    print("\nChecking optional dependencies:")
    for dep in optional_deps:
        if check_module(dep):
            print(f"✓ {dep} imported successfully")
        else:
            print(f"! {dep} import failed (optional)")

    if all_passed:
        print("\nAll core modules and dependencies imported successfully!")
        return 0
    else:
        print("\nSome imports failed. Please check your installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
