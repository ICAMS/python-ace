#!/usr/bin/env python3
"""
Compatibility test for pyace package across Python 3.9-3.13
Automatically creates a temporary venv for testing.

Usage:
    /opt/homebrew/bin/python3.9 test_compatibility.py
    /opt/homebrew/bin/python3.10 test_compatibility.py
    /opt/homebrew/bin/python3.11 test_compatibility.py
    /opt/homebrew/bin/python3.12 test_compatibility.py
    /opt/homebrew/bin/python3.13 test_compatibility.py
"""

import sys
import os
import platform
import subprocess
import tempfile
import shutil
from pathlib import Path

def get_version(module, name):
    """Safely get version of a module."""
    version_attrs = ['__version__', '_version', 'version', 'VERSION']
    for attr in version_attrs:
        if hasattr(module, attr):
            version = getattr(module, attr)
            if callable(version):
                version = version()
            if hasattr(version, '__version__'):
                version = version.__version__
            return str(version)
    return "unknown version"

def run_in_venv(venv_python, code):
    """Run Python code in the virtual environment."""
    result = subprocess.run(
        [venv_python, '-c', code],
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout, result.stderr

def test_imports_in_venv(venv_python):
    """Test imports within the virtual environment."""
    test_code = '''
import sys
import platform

def get_version(module, name):
    """Safely get version of a module."""
    version_attrs = ["__version__", "_version", "version", "VERSION"]
    for attr in version_attrs:
        if hasattr(module, attr):
            version = getattr(module, attr)
            if callable(version):
                version = version()
            if hasattr(version, "__version__"):
                version = version.__version__
            return str(version)
    return "unknown version"

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

# Test numpy
try:
    import numpy as np
    print(f"✓ numpy {get_version(np, 'numpy')}")
except ImportError as e:
    print(f"✗ numpy: {e}")
    sys.exit(1)

# Test pandas  
try:
    import pandas as pd
    print(f"✓ pandas {get_version(pd, 'pandas')}")
except ImportError as e:
    print(f"✗ pandas: {e}")
    sys.exit(1)

# Test ase
try:
    import ase
    print(f"✓ ase {get_version(ase, 'ase')}")
except ImportError as e:
    print(f"✗ ase: {e}")
    sys.exit(1)

# Test ruamel.yaml
try:
    import ruamel.yaml
    print(f"✓ ruamel.yaml imported")
except ImportError as e:
    print(f"✗ ruamel.yaml: {e}")
    sys.exit(1)

# Test scikit-learn
try:
    import sklearn
    print(f"✓ scikit-learn {get_version(sklearn, 'sklearn')}")
except ImportError as e:
    print(f"✗ scikit-learn: {e}")
    sys.exit(1)

# Test packaging for Python 3.12+
if sys.version_info >= (3, 12):
    try:
        import packaging
        print(f"✓ packaging {get_version(packaging, 'packaging')}")
    except ImportError as e:
        print(f"✗ packaging: {e}")
        sys.exit(1)

# Test pyace
try:
    import pyace
    print(f"✓ pyace {get_version(pyace, 'pyace')}")
    from pyace.basis import BBasisConfiguration
    print("✓ pyace.basis imports")
    from pyace.atomicenvironment import ACEAtomicEnvironment
    print("✓ pyace.atomicenvironment imports")
except ImportError as e:
    print(f"✗ pyace: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ pyace test failed: {e}")
    sys.exit(1)

print("✅ All imports successful!")
'''
    
    success, stdout, stderr = run_in_venv(venv_python, test_code)
    if stdout:
        print(stdout)
    if stderr and not success:
        print(f"Errors:\n{stderr}", file=sys.stderr)
    return success

def create_and_test_venv():
    """Create a temporary venv and run tests."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print("=" * 60)
    print(f"PyACE Compatibility Test - Python {python_version}")
    print("=" * 60)
    print(f"Using Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Get the pyace source directory (where this script is located)
    script_dir = Path(__file__).parent.absolute()
    
    # Create temporary directory for venv
    with tempfile.TemporaryDirectory(prefix=f"pyace_test_py{python_version}_") as temp_dir:
        venv_dir = Path(temp_dir) / "venv"
        print(f"\n📁 Creating temporary venv in: {venv_dir}")
        
        # Create virtual environment
        print("📦 Creating virtual environment...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to create venv: {result.stderr}")
            return 1
        
        # Determine venv Python executable
        if platform.system() == "Windows":
            venv_python = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
        else:
            venv_python = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"
        
        # Upgrade pip
        print("📦 Upgrading pip...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            check=False
        )
        
        # Install dependencies
        print("📦 Installing dependencies...")
        deps = ["numpy", "pandas", "ase", "ruamel.yaml", "scikit-learn", "psutil"]
        if sys.version_info >= (3, 12):
            deps.append("packaging")
        
        for dep in deps:
            print(f"   Installing {dep}...")
            result = subprocess.run(
                [str(pip_exe), "install", dep],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"   ⚠️  Warning: Failed to install {dep}")
                print(f"      {result.stderr}")
        
        # Install pyace in editable mode
        print(f"📦 Installing pyace from {script_dir}...")
        result = subprocess.run(
            [str(pip_exe), "install", "-e", str(script_dir)],
            capture_output=True,
            text=True,
            cwd=str(script_dir)
        )
        if result.returncode != 0:
            print(f"❌ Failed to install pyace:")
            print(result.stderr)
            return 1
        
        # Run tests
        print("\n🧪 Running import tests...")
        print("-" * 40)
        
        if test_imports_in_venv(venv_python):
            print("-" * 40)
            print(f"✅ Python {python_version} compatibility test PASSED!")
            return 0
        else:
            print("-" * 40)
            print(f"❌ Python {python_version} compatibility test FAILED!")
            return 1

def main():
    """Main entry point."""
    # Check if running directly (not in a venv created by this script)
    if os.environ.get('PYACE_TEST_VENV') == '1':
        # We're inside the test venv, just run the imports
        import numpy as np
        import pandas as pd
        import ase
        import ruamel.yaml
        import sklearn
        if sys.version_info >= (3, 12):
            import packaging
        import pyace
        print("✅ All imports successful in venv!")
        return 0
    
    # Check Python version
    if sys.version_info < (3, 9):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} is not supported.")
        print("   PyACE requires Python 3.9 or later.")
        return 1
    
    if sys.version_info >= (3, 14):
        print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor} has not been tested.")
        print("   PyACE is tested with Python 3.9-3.13.")
    
    # Run the venv test
    return create_and_test_venv()

if __name__ == "__main__":
    sys.exit(main())
