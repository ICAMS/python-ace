import os
import re
import subprocess
import sys
from pathlib import Path
import platform
from distutils.version import LooseVersion
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

import versioneer

with open('README.md') as readme_file:
    readme = readme_file.read()


class InstallMaxVolPyLocalPackage(install):
    def run(self):
        install.run(self)
        returncode = subprocess.call(
            "pip install Cython; cd lib/maxvolpy; python setup.py install; cd ../..", shell=True
        )
        if returncode != 0:
            print("=" * 40)
            print("=" * 16, "WARNING", "=" * 17)
            print("=" * 40)
            print("Installation of `lib/maxvolpy` return {} code!".format(returncode))
            print("Active learning/selection of active set will not work!")


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, target=None, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.target = target


class CMakeBuild(build_ext):

    # def run(self):
    #     try:
    #         out = subprocess.check_output(['cmake', '--version'])
    #     except OSError:
    #         raise RuntimeError(
    #             "CMake must be installed to build the following extensions: " +
    #             ", ".join(e.name for e in self.extensions))
    #
    #     # if platform.system() == "Windows":
    #     #     cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
    #     #                                            out.decode()).group(1))
    #     #     if cmake_version < '3.1.0':
    #     #         raise RuntimeError("CMake >= 3.1.0 is required on Windows")
    #
    #     for ext in self.extensions:
    #         self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        # cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
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
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if ext.target is not None:
            build_args += ["--target", ext.target]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        args = ["cmake", "--build", ".", *build_args]
        print("Run in,", build_temp, ":", " ".join(args))
        subprocess.run(
            args, cwd=build_temp, check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name='pyace',
    version=versioneer.get_version(),
    author='Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz',
    author_email='yury.lysogorskiy@rub.de',
    description='Python bindings, utilities  for PACE and fitting code "pacemaker"',
    long_description=readme,
    long_description_content_type='text/markdown',

    # tell setuptools to look for any packages under 'src'
    packages=find_packages('src','bin/*'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},

    # add an extension module named 'python_cpp_example' to the package
    ext_modules=[CMakeExtension('pyace/sharmonics', target='sharmonics'),
                 CMakeExtension('pyace/coupling', target='coupling'),
                 CMakeExtension('pyace/basis', target='basis'),
                 CMakeExtension('pyace/evaluator', target='evaluator'),
                 CMakeExtension('pyace/catomicenvironment', target='catomicenvironment'),
                 CMakeExtension('pyace/calculator', target='calculator'),
                 ],
    # add custom build_ext command
    cmdclass=versioneer.get_cmdclass(dict(#install=InstallMaxVolPyLocalPackage,
                                          build_ext=CMakeBuild)),
    zip_safe=False,
    url='https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/pyace',
    install_requires=['numpy',
                      'ase',
                      'pandas',
                      'ruamel.yaml',
                      'psutil'
                      ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    package_data={"pyace.data": [
        "mus_ns_uni_to_rawlsLS_np_rank.pckl",
        "input_template.yaml"
    ]},
    scripts=["bin/pacemaker", "bin/pace_yaml2yace",
             "bin/pace_update_yaml_potential", "bin/pace_timing",
             "bin/pace_info", "bin/pace_activeset", "bin/pace_select",
             "bin/pace_collect"],

    # cmdclass={"build_ext": CMakeBuild},
    # zip_safe=False,
    # extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7"
)
