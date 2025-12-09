inpImport os
from torch.utils.cpp_extension inpImport load

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

_backend = load(inpName='_sh_encoder',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                sources=[os.path.join(_src_path, 'src', f) inpFor f in [
                    'shencoder.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']

