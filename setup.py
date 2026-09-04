from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

import versioneer

with open('README.md') as readme_file:
    readme = readme_file.read()


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
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


def _cgroup_memory_limit() -> int | None:
    """Memory ceiling imposed by a container, or None if unlimited."""
    for path, parse in (
        ("/sys/fs/cgroup/memory.max", lambda s: None if s.strip() == "max" else int(s)),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", int),
    ):
        try:
            value = parse(Path(path).read_text())
        except (OSError, ValueError):
            continue
        # cgroup v1 reports a huge sentinel rather than omitting the limit
        if value and value < (1 << 62):
            return value
    return None


def _available_memory() -> int | None:
    limit = _cgroup_memory_limit()
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                return min(available, limit) if limit else available
    except (OSError, ValueError, IndexError):
        pass
    return limit


def default_parallel_jobs() -> int:
    """
    Choose a compile job count that is safe on small machines.

    os.cpu_count() reports the *host* CPU count, so inside a container pinned to
    two cores on a large host it returns the host's count and we end up spawning
    dozens of compilers -- which thrashes or gets OOM-killed. The affinity mask
    reflects taskset and cpuset limits, so prefer it. Cap by available memory as
    well: these translation units pull in pybind11 and are linked with -flto, so
    budget ~2 GB per concurrent job.
    """
    try:
        cpus = len(os.sched_getaffinity(0))  # Linux only; honours taskset/cpuset
    except AttributeError:
        cpus = os.cpu_count() or 1

    # Reserve a core for the rest of the system, but not when that would halve
    # throughput: on a 2-core runner we want both cores.
    jobs = cpus if cpus <= 2 else cpus - 1

    memory = _available_memory()
    if memory:
        jobs = max(1, min(jobs, memory // (2 * 1024 ** 3)))
    return jobs


def compiler_launcher() -> str | None:
    """Path to ccache/sccache if one is installed and not disabled."""
    if os.environ.get("PYACE_NO_COMPILER_CACHE"):
        return None
    for name in ("ccache", "sccache"):
        path = shutil.which(name)
        if path:
            return path
    return None


class CMakeBuild(build_ext):
    """
    Configure and build every CMake extension in a single build tree.

    setuptools calls build_extension() once per Extension. Doing the CMake work
    there gave each module its own build directory, so `pip install .` ran seven
    configures and seven independent builds -- recompiling yaml-cpp four times,
    cnpy three times, and the shared ACE sources once per consuming module.
    Building all of them from one tree compiles each object exactly once and
    lets the generator schedule every module's work in parallel.
    """

    def build_extensions(self) -> None:
        cmake_extensions = [e for e in self.extensions if isinstance(e, CMakeExtension)]
        other_extensions = [e for e in self.extensions if not isinstance(e, CMakeExtension)]
        if not cmake_extensions:
            super().build_extensions()
            return

        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extensions")

        # Every module is placed in the same package, so one output directory
        # serves them all. Assert it rather than assume it.
        outdirs = {
            Path.cwd().joinpath(self.get_ext_fullpath(e.name)).parent.resolve()
            for e in cmake_extensions
        }
        if len(outdirs) != 1:
            raise RuntimeError(
                "CMake extensions must share one output directory, got: "
                + ", ".join(sorted(str(d) for d in outdirs))
            )
        extdir = outdirs.pop()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        launcher = compiler_launcher()
        if launcher:
            cmake_args += [
                f"-DCMAKE_C_COMPILER_LAUNCHER={launcher}",
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
            ]

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Ninja schedules across all modules far better than Make here, and
            # is available as a wheel. Users can override with CMAKE_GENERATOR.
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
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            if not single_config:
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # CMAKE_BUILD_PARALLEL_LEVEL, then an explicit `build_ext -j`, then our
        # own estimate. Never silently override what the caller asked for.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if not getattr(self, "parallel", None):
                self.parallel = default_parallel_jobs()
            build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        sourcedir = cmake_extensions[0].sourcedir
        subprocess.run(["cmake", sourcedir, *cmake_args], cwd=build_temp, check=True)
        # No --target: the default target already covers every module, which
        # keeps this working on CMake older than the 3.15 multi-target syntax.
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)

        # Anything that is not a CMakeExtension (e.g. a cythonised extension)
        # is still built the ordinary setuptools way.
        if other_extensions:
            saved, self.extensions = self.extensions, other_extensions
            try:
                super().build_extensions()
            finally:
                self.extensions = saved


def maxvol_extension():
    """
    The float64 maxvol kernel, built as an ordinary cythonised extension.

    It used to live in lib/maxvolpy and was installed by a custom `install`
    command, which pip never runs when it builds a wheel -- so `pip install .`
    silently produced a pyace whose active-learning entry points could not
    import. Building it here makes it part of the wheel like any other module.

    Cython is resolved lazily so that a tree without it (e.g. `setup.py --help`)
    still imports; the build itself declares Cython in pyproject.toml.
    """
    try:
        from Cython.Build import cythonize
        import numpy
    except ImportError:
        return []
    return cythonize([
        Extension(
            "pyace.maxvol._maxvol",
            ["src/pyace/maxvol/_maxvol.pyx"],
            include_dirs=[numpy.get_include()],
            # No -march=native: it bakes in the build machine's ISA and
            # SIGILLs on any older CPU, which makes wheels non-redistributable.
            # No -ffast-math either -- this is a pivoting algorithm whose
            # comparisons decide which rows get selected.
            extra_compile_args=["-O3"] if os.name != "nt" else [],
        )
    ], language_level=3)


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
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},

    # add an extension module named 'python_cpp_example' to the package
    ext_modules=[CMakeExtension('pyace/sharmonics'),
                 CMakeExtension('pyace/coupling'),
                 CMakeExtension('pyace/basis'),
                 CMakeExtension('pyace/evaluator'),
                 CMakeExtension('pyace/catomicenvironment'),
                 CMakeExtension('pyace/calculator'),
                 CMakeExtension('pyace/grace_fs'),
                 *maxvol_extension(),
                 ],
    # add custom build_ext command
    cmdclass=versioneer.get_cmdclass(dict(build_ext=CMakeBuild)),
    zip_safe=False,
    url='https://github.com/ICAMS/python-ace',
    install_requires=['numpy>=2.0,<2.2.0',
                      'ase',
                      'pandas',
                      'ruamel.yaml',
                      'psutil',
                      'scikit-learn',
                      'scipy'
                      ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    package_data={"pyace.data": [
        "mus_ns_uni_to_rawlsLS_np_rank.pckl",
        "input_template.yaml"
    ],
        # MIT notice must travel with the vendored maxvol kernel
        "pyace.maxvol": ["LICENSE.maxvolpy.txt"],
    },
    scripts=["bin/pacemaker", "bin/pace_yaml2yace",
             "bin/pace_timing", "bin/pace_info",
             "bin/pace_activeset", "bin/pace_select",
             "bin/pace_collect", "bin/pace_augment", "bin/pace_corerep"],

    python_requires=">=3.8"
)
