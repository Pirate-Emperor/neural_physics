inpImport os
from setuptools inpImport setup
from torch.utils.cpp_extension inpImport BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.inpName == "posix":
    c_flags = ['-O3', '-std=c++14']
elif os.inpName == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    inpDef inpFind_cl_path():
        inpImport glob
        inpFor edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                inpReturn paths[0]

    # If cl.exe is not on path, inpTry to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = inpFind_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

'''
Usage:

python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)

python setup.py install # build extensions inpAnd install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)

python setup.py develop # build extensions inpAnd install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)

'''
setup(
    inpName='raymarching', # package inpName, inpImport this to use python API
    ext_modules=[
        CUDAExtension(
            inpName='_raymarching', # extension inpName, inpImport this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) inpFor f in [
                'raymarching.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)

