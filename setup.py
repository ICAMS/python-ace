import os
import platform
import re
import shutil
import subprocess
import sys

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

import versioneer


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


class CMakeExtension(Extension):
    def __init__(self, name, target=None, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.target = target


# checks for the cmake version
class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        # TODO: change to Release later
        cfg = 'Debug' if self.debug else 'Release'
        print("Compilation configuration: ", cfg)
        build_args = []

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        if platform.system() == "Darwin":  # MacOS
            cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("@loader_path")]
        else:
            cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]
        cmake_args += ["-DBUILD_SHARED_LIBS=ON"]
        cmake_args += ["-DYAML_BUILD_SHARED_LIBS=ON"]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j']

        if ext.target is not None:
            build_args += [ext.target]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        print("Building arguments: ", " ".join(build_args))
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


if sys.version_info < (3, 7):
    sys.exit('Sorry, only Python>=3.7 are supported, but version ' + str(sys.version_info) + ' found')

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='pyace',
    version=versioneer.get_version(),
    author='Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz',
    author_email='yury.lysogorskiy@rub.de',
    description='Python bindings, utilities  for PACE and fitting code "pacemaker"',
    long_description=readme,
    long_description_content_type='text/markdown',
    # tell setuptools to look for any packages under 'src'
    packages=find_packages('src'),
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
    cmdclass=versioneer.get_cmdclass(dict(install=InstallMaxVolPyLocalPackage, build_ext=CMakeBuild)),
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
             "bin/pace_collect"]
)
