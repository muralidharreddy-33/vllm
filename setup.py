# SPDX-License-Identifier: Apache-2.0

import ctypes
import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Dict, List

import torch
from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

# cannot import envs directly because it depends on vllm,
#  which is not installed yet
envs = load_module_from_path('envs', os.path.join(ROOT_DIR, 'vllm', 'envs.py'))

VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE

if sys.platform.startswith("darwin") and VLLM_TARGET_DEVICE != "cpu":
    logger.warning(
        "VLLM_TARGET_DEVICE automatically set to `cpu` due to macOS")
    VLLM_TARGET_DEVICE = "cpu"
elif not (sys.platform.startswith("linux")
          or sys.platform.startswith("darwin")):
    logger.warning(
        "vLLM only supports Linux platform (including WSL) and MacOS."
        "Building on %s, "
        "so vLLM may not be able to run correctly", sys.platform)
    VLLM_TARGET_DEVICE = "empty"
elif (sys.platform.startswith("linux") and torch.version.cuda is None
      and torch.version.hip is None
      and os.getenv("VLLM_TARGET_DEVICE") is None):
    # if cuda is not available and VLLM_TARGET_DEVICE is not set,
    # fallback to cpu
    VLLM_TARGET_DEVICE = "cpu"

MAIN_CUDA_VERSION = "12.1"


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = envs.NVCC_THREADS
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info(
                    "Using NVCC_THREADS=%d as the number of nvcc threads.",
                    nvcc_threads)
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DVLLM_TARGET_DEVICE={}'.format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DVLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm.").removeprefix("vllm_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            # We assume only the final component of extension prefix is added by
            # CMake, this is currently true for current extensions but may not
            # always be the case.
            prefix = outdir
            if '.' in ext.name:
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake", "--install", ".", "--prefix", prefix, "--component",
                target_name(ext.name)
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()

        # copy vllm/vllm_flash_attn/*.py from self.build_lib to current
        # directory so that they can be included in the editable build
        import glob
        files = glob.glob(
            os.path.join(self.build_lib, "vllm", "vllm_flash_attn", "*.py"))
        for file in files:
            dst_file = os.path.join("vllm/vllm_flash_attn",
                                    os.path.basename(file))
            print(f"Copying {file} to {dst_file}")
            self.copy_file(file, dst_file)


class repackage_wheel(build_ext):
    """Extracts libraries and other files from an existing wheel."""
    default_wheel = "https://wheels.vllm.ai/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

    def run(self) -> None:
        wheel_location = os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION",
                                   self.default_wheel)

        assert _is_cuda(
        ), "VLLM_USE_PRECOMPILED is only supported for CUDA builds"

        import zipfile

        if os.path.isfile(wheel_location):
            wheel_path = wheel_location
            print(f"Using existing wheel={wheel_path}")
        else:
            # Download the wheel from a given URL, assume
            # the filename is the last part of the URL
            wheel_filename = wheel_location.split("/")[-1]

            import tempfile

            # create a temporary directory to store the wheel
            temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
            wheel_path = os.path.join(temp_dir, wheel_filename)

            print(f"Downloading wheel from {wheel_location} to {wheel_path}")

            from urllib.request import urlretrieve

            try:
                urlretrieve(wheel_location, filename=wheel_path)
            except Exception as e:
                from setuptools.errors import SetupError

                raise SetupError(
                    f"Failed to get vLLM wheel from {wheel_location}") from e

        with zipfile.ZipFile(wheel_path) as wheel:
            files_to_copy = [
                "vllm/_C.abi3.so",
                "vllm/_moe_C.abi3.so",
                "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
                "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
                "vllm/vllm_flash_attn/flash_attn_interface.py",
                "vllm/vllm_flash_attn/__init__.py",
                "vllm/cumem_allocator.abi3.so",
                # "vllm/_version.py", # not available in nightly wheels yet
            ]
            file_members = filter(lambda x: x.filename in files_to_copy,
                                  wheel.filelist)

            for file in file_members:
                print(f"Extracting and including {file.filename} "
                      "from existing wheel")
                package_name = os.path.dirname(file.filename).replace("/", ".")
                file_name = os.path.basename(file.filename)

                if package_name not in package_data:
                    package_data[package_name] = []

                wheel.extract(file)
                if file_name.endswith(".py"):
                    # python files shouldn't be added to package_data
                    continue

                package_data[package_name].append(file_name)


def _is_hpu() -> bool:
    # if VLLM_TARGET_DEVICE env var was set explicitly, skip HPU autodetection
    if os.getenv("VLLM_TARGET_DEVICE", None) == VLLM_TARGET_DEVICE:
        return VLLM_TARGET_DEVICE == "hpu"

    # if VLLM_TARGET_DEVICE was not set explicitly, check if hl-smi succeeds,
    # and if it doesn't, check if habanalabs driver is loaded
    is_hpu_available = False
    try:
        out = subprocess.run(["hl-smi"], capture_output=True, check=True)
        is_hpu_available = out.returncode == 0
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        if sys.platform.startswith("linux"):
            try:
                output = subprocess.check_output(
                    'lsmod | grep habanalabs | wc -l', shell=True)
                is_hpu_available = int(output) > 0
            except (ValueError, FileNotFoundError, PermissionError,
                    subprocess.CalledProcessError):
                pass
    return is_hpu_available


def _no_device() -> bool:
    return VLLM_TARGET_DEVICE == "empty"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return (VLLM_TARGET_DEVICE == "cuda" and has_cuda
            and not (_is_neuron() or _is_tpu() or _is_hpu()))


def _is_hip() -> bool:
    return (VLLM_TARGET_DEVICE == "cuda"
            or VLLM_TARGET_DEVICE == "rocm") and torch.version.hip is not None


def _is_neuron() -> bool:
    return VLLM_TARGET_DEVICE == "neuron"


def _is_tpu() -> bool:
    return VLLM_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return VLLM_TARGET_DEVICE == "cpu"


def _is_openvino() -> bool:
    return VLLM_TARGET_DEVICE == "openvino"


def _is_xpu() -> bool:
    return VLLM_TARGET_DEVICE == "xpu"


def _build_custom_ops() -> bool:
    return _is_cuda() or _is_hip() or _is_cpu()


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        VerErrors = ctypes.c_uint32
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = VerErrors
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()

        if (get_rocm_core_version(ctypes.byref(major), ctypes.byref(minor),
                                  ctypes.byref(patch)) == 0):
            return f"{major.value}.{minor.value}.{patch.value}"
        return None
    except Exception:
        return None


def get_neuronxcc_version():
    import sysconfig
    site_dir = sysconfig.get_paths()["purelib"]
    version_file = os.path.join(site_dir, "neuronxcc", "version",
                                "__init__.py")

    # Check if the command was executed successfully
    with open(version_file) as fp:
        content = fp.read()

    # Extract the version using a regular expression
    match = re.search(r"__version__ = '(\S+)'", content)
    if match:
        # Return the version string
        return match.group(1)
    else:
        raise RuntimeError("Could not find Neuron version in the output")


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_gaudi_sw_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run("hl-smi",
                            shell=True,
                            text=True,
                            capture_output=True,
                            env={"ENABLE_CONSOLE": "true"})
    if output.returncode == 0 and output.stdout:
        return output.stdout.split("\n")[2].replace(
            " ", "").split(":")[1][:-1].split("-")[0]
    return "0.0.0"  # when hl-smi is not available


def get_vllm_version() -> str:
    version = get_version(
        write_to="vllm/_version.py",  # TODO: move this to pyproject.toml
    )
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _no_device():
        if envs.VLLM_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
        if envs.VLLM_USE_PRECOMPILED:
            version += f"{sep}precompiled"
        else:
            cuda_version = str(get_nvcc_cuda_version())
            if cuda_version != MAIN_CUDA_VERSION:
                cuda_version_str = cuda_version.replace(".", "")[:3]
                # skip this for source tarball, required for pypi
                if "sdist" not in sys.argv:
                    version += f"{sep}cu{cuda_version_str}"
    elif _is_hip():
        # Get the Rocm Version
        rocm_version = get_rocm_version() or torch.version.hip
        if rocm_version and rocm_version != MAIN_CUDA_VERSION:
            version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    elif _is_neuron():
        # Get the Neuron version
        neuron_version = str(get_neuronxcc_version())
        if neuron_version != MAIN_CUDA_VERSION:
            neuron_version_str = neuron_version.replace(".", "")[:3]
            version += f"{sep}neuron{neuron_version_str}"
    elif _is_hpu():
        # Get the Intel Gaudi Software Suite version
        gaudi_sw_version = str(get_gaudi_sw_version())
        if gaudi_sw_version != MAIN_CUDA_VERSION:
            gaudi_sw_version = gaudi_sw_version.replace(".", "")[:3]
            version += f"{sep}gaudi{gaudi_sw_version}"
    elif _is_openvino():
        version += f"{sep}openvino"
    elif _is_tpu():
        version += f"{sep}tpu"
    elif _is_cpu():
        if envs.VLLM_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    elif _is_xpu():
        version += f"{sep}xpu"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    if _no_device():
        requirements = _read_requirements("requirements-common.txt")
    elif _is_cuda():
        requirements = _read_requirements("requirements-cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if ("vllm-flash-attn" in req
                    and not (cuda_major == "12" and cuda_minor == "1")):
                # vllm-flash-attn is built only for CUDA 12.1.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements
    elif _is_hip():
        requirements = _read_requirements("requirements-rocm.txt")
    elif _is_neuron():
        requirements = _read_requirements("requirements-neuron.txt")
    elif _is_hpu():
        requirements = _read_requirements("requirements-hpu.txt")
    elif _is_openvino():
        requirements = _read_requirements("requirements-openvino.txt")
    elif _is_tpu():
        requirements = _read_requirements("requirements-tpu.txt")
    elif _is_cpu():
        requirements = _read_requirements("requirements-cpu.txt")
    elif _is_xpu():
        requirements = _read_requirements("requirements-xpu.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, Neuron, HPU, "
            "OpenVINO, or CPU.")
    return requirements


ext_modules = []

if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._moe_C"))

if _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._rocm_C"))

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))
    if envs.VLLM_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.0"):
        # FA3 requires CUDA 12.0 or later
        ext_modules.append(
            CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
    ext_modules.append(CMakeExtension(name="vllm.cumem_allocator"))

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="vllm._C"))

package_data = {
    "vllm": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
    ]
}

if _no_device():
    ext_modules = []

if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {
        "build_ext":
        repackage_wheel if envs.VLLM_USE_PRECOMPILED else cmake_build_ext
    }

setup(
    name="vllm",
    version=get_vllm_version(),
    author="vLLM Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
        "Documentation": "https://vllm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples",
                                    "tests*")),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    extras_require={
        "tensorizer": ["tensorizer>=2.9.0"],
        "runai": ["runai-model-streamer", "runai-model-streamer-s3", "boto3"],
        "audio": ["librosa", "soundfile"],  # Required for audio processing
        "video": ["decord"]  # Required for video processing
    },
    cmdclass=cmdclass,
    package_data=package_data,
    entry_points={
        "console_scripts": [
            "vllm=vllm.scripts:main",
        ],
    },
)
