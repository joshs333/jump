# JUMP CMake
There are a series of cmake functions that are used to help create targets with certain backends.

## CUDA
The following functions are available:

**Enable CUDA Backend on existing target**
Link all required libraries and define all flags to a target to enable the CUDA backend, also enables the CUDA langage and separable compilation so .cu files are compiled with nvcc.
```
jump_cuda_target(<target_name>)
```

**Mark .cpp files as .cu files at cmake**
I tend to favor the .cpp and .hpp extensions for all code, with the implicit assumption that if you are not using the CUDA backend then the header files will have the CUDA specific parts removed, and conversely if the CUDA backend is enabled then some CUDA specific code is included that will require the source be compiled with nvcc.

To facilitate this, if I am compiling with the CUDA backend, I use a cmake macro to mark .cpp sources as the CUDA language to force it to compile those files as though they were .cu files.
```
jump_cuda_sources(<sources> ...)
```

To use this macro (due to some cmake internals I don't quite understand, since to my understanding it should work with just enable_language(CUDA)), you must have CUDA declared as a language in the project declaration. Eg:
```
project(my_project VERSION 0.1 LANGUAGES CXX CUDA)
```
Note that the use of this macro is the only reason you should need to mark a language besides CXX in the project declaration, all other macros should automatically enable the languages they need. Alternatively, make your sources .cu and .cpp to appropriately mark which sources are CUDA or not.
