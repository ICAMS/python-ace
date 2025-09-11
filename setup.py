import os
import re
import subprocess
import sys
from pathlib import Path
import platform

# Handle distutils removal in Python 3.12+
try:
    from distutils.version import LooseVersion
except ImportError:
    from packaging.version import Version as LooseVersion

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


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
    """A CMakeExtension needs a sourcedir instead of a file list."""
    
    def __init__(self, name: str, target=None, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.target = target


class CMakeBuild(build_ext):
    """Custom build extension for CMake-based builds."""

    def build_extension(self, ext: CMakeExtension) -> None:
        # Check if CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extensions")
        
        # Set up parallel build
        self.parallel = os.cpu_count() - 1
        if self.parallel < 1:
            self.parallel = 1
            
        # Get extension paths
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Build configuration
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake generator
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set up CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = []
        
        # Add environment CMake arguments
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Handle different compilers and generators
        if self.compiler.compiler_type != "msvc":
            # Try to use Ninja if available
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
        else:
            # Windows MSVC handling
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Add target if specified
        if ext.target is not None:
            build_args += ["--target", ext.target]

        # macOS cross-compilation support
        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        # Set parallel build level
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        # Create build directory
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Run CMake configure and build
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], 
            cwd=build_temp, 
            check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], 
            cwd=build_temp, 
            check=True
        )


# Define extensions
ext_modules = [
    CMakeExtension('pyace/sharmonics', target='sharmonics'),
    CMakeExtension('pyace/coupling', target='coupling'),
    CMakeExtension('pyace/basis', target='basis'),
    CMakeExtension('pyace/evaluator', target='evaluator'),
    CMakeExtension('pyace/catomicenvironment', target='catomicenvironment'),
    CMakeExtension('pyace/calculator', target='calculator'),
]

# Define custom commands
cmdclass = {
    'install': InstallMaxVolPyLocalPackage,
    'build_ext': CMakeBuild,
}

# Try to get version from versioneer if available
try:
    import versioneer
    version = versioneer.get_version()
    cmdclass.update(versioneer.get_cmdclass())
except ImportError:
    # Fallback version
    version = "0.0.0+unknown"

# Run setup
if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        version=version,
        zip_safe=False,
    )
