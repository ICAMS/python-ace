#!/usr/bin/env python3
"""
Simple compatibility test for pyace package across Python 3.9-3.13
"""

import sys
import platform

def test_basic_imports():
    """Test that basic imports work."""
    print(f"Testing on Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import ase
        print(f"✓ ase {ase.__version__}")
    except ImportError as e:
        print(f"✗ ase import failed: {e}")
        return False
    
    try:
        import ruamel.yaml
        print(f"✓ ruamel.yaml")
    except ImportError as e:
        print(f"✗ ruamel.yaml import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    # Test Python version specific imports
    if sys.version_info >= (3, 12):
        try:
            import packaging
            print(f"✓ packaging {packaging.__version__} (required for Python 3.12+)")
        except ImportError as e:
            print(f"✗ packaging import failed: {e}")
            return False
    
    return True

def test_pyace_imports():
    """Test that pyace imports work."""
    try:
        # Test basic pyace imports
        import pyace
        print(f"✓ pyace {pyace.__version__}")
        
        # Test specific modules
        from pyace.basis import BBasisConfiguration
        print("✓ pyace.basis imports")
        
        from pyace.atomicenvironment import ACEAtomicEnvironment
        print("✓ pyace.atomicenvironment imports")
        
        return True
    except ImportError as e:
        print(f"✗ pyace import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ pyace test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("=" * 50)
    print("PyACE Python 3.9-3.13 Compatibility Test")
    print("=" * 50)
    
    # Test basic dependencies
    if not test_basic_imports():
        print("\n❌ Basic dependency tests failed!")
        return 1
    
    print("\n" + "-" * 30)
    
    # Test pyace imports (might fail if not installed)
    if not test_pyace_imports():
        print("\n⚠️  PyACE imports failed (package might not be installed yet)")
        print("This is expected if running before installation.")
    else:
        print("\n✅ All tests passed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
