"""
Setup script for WaveFactor.
Uses CMakeExtension to delegate all C++ compilation, source management,
and optimization flags to CMakeLists.txt (the single source of truth).
Based on the canonical pybind11/cmake_example architecture.
"""

import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A setuptools Extension that delegates compilation to CMake."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension that invokes CMake to compile targets."""
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(extdir, exist_ok=True)

        repo_build = os.path.join(ext.sourcedir, "build")

        # 1. If an active configured build/ directory exists, build target directly
        if os.path.isfile(os.path.join(repo_build, "CMakeCache.txt")):
            subprocess.check_call(["cmake", "--build", repo_build, "--target", ext.name])
            self._copy_built_binary(repo_build, extdir, ext.name)
            return

        # 2. Otherwise configure and build in self.build_temp
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_MODULE=ON",
        ]

        if sys.platform == "win32":
            cmake_args.extend(["-G", "MinGW Makefiles"])

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", ext.name], cwd=build_temp)
        self._copy_built_binary(build_temp, extdir, ext.name)

    def _copy_built_binary(self, src_dir, dest_dir, target_name):
        """Copies the compiled .pyd / .so extension to the target destination if needed."""
        src_dir = os.path.abspath(src_dir)
        dest_dir = os.path.abspath(dest_dir)
        if src_dir == dest_dir:
            return

        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.startswith(target_name) and (f.endswith(".pyd") or f.endswith(".so")):
                    src_file = os.path.join(root, f)
                    dest_file = os.path.join(dest_dir, f)
                    if not os.path.exists(dest_file) or os.path.getmtime(src_file) > os.path.getmtime(dest_file):
                        shutil.copy2(src_file, dest_file)
                    return


setup(
    name="wavefactor",
    version="2.0.0",
    packages=["wavefactor"],
    ext_modules=[CMakeExtension("WaveFactor")],
    cmdclass={"build_ext": CMakeBuild},
)
