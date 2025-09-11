# Python 3.9-3.13 Compatibility Changes

This document summarizes the changes made to make pyace compatible with Python 3.9 through 3.13.

## Key Changes

### 1. Modern Package Configuration
- **Added `pyproject.toml`**: Modern Python packaging standard (PEP 518/517)
- **Updated `setup.py`**: Simplified and modernized, handles CMake extensions
- **Updated `setup.cfg`**: Minimal configuration, most moved to pyproject.toml

### 2. Dependency Management
- **Updated version constraints**: Removed upper bounds that were too restrictive
- **Added conditional dependencies**: `packaging>=20.0` for Python 3.12+ (distutils removal)
- **Modernized requirements**: Specified minimum versions for better compatibility

### 3. Python Version Support
- **Supported versions**: Python 3.9, 3.10, 3.11, 3.12, 3.13
- **Build system**: Uses setuptools with CMake for C++ extensions
- **CI/CD**: Updated GitHub Actions to test all supported versions

### 4. Compatibility Fixes
- **distutils handling**: Added fallback to `packaging.version` for Python 3.12+
- **String formatting**: Code review showed no compatibility issues
- **Import statements**: All imports are compatible across versions

### 5. Development Tools
- **Added `requirements-dev.txt`**: Development dependencies
- **Added `test_compatibility.py`**: Simple test script to verify installation
- **Updated CI/CD**: Comprehensive testing across Python versions and OS

## Files Modified

### Core Configuration
- `pyproject.toml` - **NEW**: Modern package configuration
- `setup.py` - **UPDATED**: Simplified, Python 3.12+ compatible
- `setup.cfg` - **UPDATED**: Minimal configuration
- `requirements.txt` - **UPDATED**: Version constraints
- `MANIFEST.in` - **UPDATED**: Include new files

### Development
- `requirements-dev.txt` - **NEW**: Development dependencies
- `test_compatibility.py` - **NEW**: Compatibility test script
- `.github/workflows/test.yml` - **UPDATED**: Test Python 3.9-3.13

## Installation

### For Users
```bash
pip install .
```

### For Developers
```bash
pip install -r requirements-dev.txt
pip install -e .
```

### Testing Compatibility
```bash
python test_compatibility.py
```

## Key Dependencies

- **numpy**: ≥1.19.0 (Python 3.9+ compatible)
- **ase**: ≥3.22.0 (Atomic Simulation Environment)
- **pandas**: ≥1.3.0 (Data manipulation)
- **scikit-learn**: ≥1.0.0 (Machine learning)
- **packaging**: ≥20.0 (Python 3.12+ only, replaces distutils)

## Build Requirements

- **CMake**: ≥3.12
- **C++ compiler**: C++14 compatible
- **pybind11**: ≥2.6.0
- **ninja**: Recommended for faster builds

## Testing

The package is tested on:
- **Python versions**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Operating systems**: Ubuntu, macOS
- **Architectures**: x86_64, ARM64 (Apple Silicon)

## Backward Compatibility

All changes maintain backward compatibility with existing:
- API interfaces
- Configuration files
- Data formats
- Usage patterns

## Future Maintenance

- Monitor Python release schedule for new versions
- Update CI/CD when Python 3.14 is released
- Review dependencies for compatibility issues
- Consider migration to newer build systems (e.g., scikit-build-core) when appropriate
