"""
Modern setup.py for pyace package with proper CMake integration.
All metadata is in pyproject.toml. This file handles CMake extensions properly.
"""

import os
import re
import subprocess
import sys
import shutil
from pathlib import Path
import platform

# Handle distutils removal in Python 3.12+
try:
    from distutils.version import LooseVersion
except ImportError:
    from packaging.version import Version as LooseVersion

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# Import versioneer
import versioneer


class InstallMaxVolPyLocalPackage(install):
    """Custom install command to handle maxvolpy dependency."""
    
    def run(self):
        install.run(self)
        cmd = "cd lib/maxvolpy; python setup.py install; cd ../.."
        if platform.system() != "Windows":
            cmd = "pip install Cython; " + cmd
        returncode = subprocess.call(cmd, shell=True)
        if returncode != 0:
            print("=" * 40)
            print("=" * 16, "WARNING", "=" * 17)
            print("=" * 40)
            print(f"Installation of `lib/maxvolpy` returned {returncode} code!")
            print("Active learning/selection of active set will not work!")


# Convert Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64", 
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    """Extension that uses CMake to build."""
    
    def __init__(self, name, target=None, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.target = target


class CMakeBuild(build_ext):
    """Build extension using CMake."""
    
    def run(self):
        """Run the build process."""
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        # Call parent to set up compiler
        super().run()

    def build_extension(self, ext):
        """Build a single extension using CMake."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = []

        # Adding CMake arguments set as environment variable
        if "CMAKE_ARGS" in os.environ:
            import shlex
            cmake_args += shlex.split(os.environ["CMAKE_ARGS"])

        # Set up build parallelism
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        # Handle ninja generator
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja
                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        if ext.target is not None:
            build_args += ["--target", ext.target]

        # Cross-compile support for macOS
        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


# Define extensions
ext_modules = [
    CMakeExtension('pyace.sharmonics', target='sharmonics'),
    CMakeExtension('pyace.coupling', target='coupling'), 
    CMakeExtension('pyace.basis', target='basis'),
    CMakeExtension('pyace.evaluator', target='evaluator'),
    CMakeExtension('pyace.catomicenvironment', target='catomicenvironment'),
    CMakeExtension('pyace.calculator', target='calculator'),
]

# Set up command classes
cmdclass = versioneer.get_cmdclass()
cmdclass.update({
    'install': InstallMaxVolPyLocalPackage,
    'build_ext': CMakeBuild,
})

# Run setup with full configuration since pyproject.toml only has build info
if __name__ == "__main__":
    setup(
        name="pyace",
        version=versioneer.get_version(),
        author="Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz",
        author_email="yury.lysogorskiy@rub.de",
        description="Python bindings, utilities for PACE and fitting code 'pacemaker'",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url="https://github.com/ICAMS/python-ace",
        packages=find_packages('src'),
        package_dir={'': 'src'},
        python_requires=">=3.9,<3.14",
        install_requires=[
            "numpy>=1.19.0",
            "ase>=3.22.0", 
            "pandas>=1.3.0",
            "ruamel.yaml>=0.15.0",
            "psutil>=5.0.0",
            "scikit-learn>=1.0.0",
            "packaging>=20.0; python_version>='3.12'",
        ],
        license="Apache-2.0",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Chemistry",
        ],
        package_data={
            "pyace.data": ["*.pckl", "*.yaml", "*.gzip"],
            "pyace": ["py.typed"],
        },
        entry_points={
            "console_scripts": [
                "pacemaker=pyace.cli:pacemaker_main",
            ],
        },
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        zip_safe=False,
        # Keep the original scripts available
        scripts=[
            "bin/pacemaker",
            "bin/pace_yaml2yace", 
            "bin/pace_timing",
            "bin/pace_info",
            "bin/pace_activeset",
            "bin/pace_select",
            "bin/pace_collect",
            "bin/pace_augment",
            "bin/pace_corerep",
        ],
    )
