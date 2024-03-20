# High-Level Design
## Interopable Data Structures
A large design goal for this library was interopable data structures to allow code to run in parallel on GPU or CPU without funky templating, raw pointers, or code re-writing with different types.

We support the following (memory) data structures:
- a shared pointer (jump::shared_ptr) that can be transfered from CPU to GPU
- a sequential array (jump::array)
- a multi-dimensional array (up to 4 dimensions by default) (jump::multi_array)
- the backend for both of these is an interopable memory buffer (jump::memory_buffer)

We also support:
- interopable (same code / types) random number generation
- string_view (to suppport constexpr string comparison on GPU)
- string (to support non-constexpr string operations)
    GOTCHA! -> string does not support direct passing to kernel because of ownership / copy semantics of the string, see src/tests/string.cpp


Other functionality:
- run-time selection of parallelism targets (threadpool, GPU, sequential)
- yaml read / write for array's

# Open Questions / To-Do's
- Fix CPU float atomic operations
- should I have made the kernels / executors const to be passed to the iterate / foreach calls (same for threadpool executor interfaces)
- make foreach / iterate callable from device??
- make cuda calls for multi_array use 3d blocks or whatevs? (easy to hit limit with three dimensional array)
- flesh out unit tests for parallel, threadpool, new multi_array features with the indices
- provide explicit clone / copy methods for array and memory_array
- support device-only arrays, multi-arrays, etc... just won't be able to do class constructor initialization.
- flesh out API more extensively (can do more constexpr, can do more std library string functions / etc...)
- string always does sole ownership of buffer - memory_buffer does reference counting which is uneeded overhead for string (add unshared memory_buffer type?)
- interopable view? -> view has resolved device / host pointers allowing it to be passed between cpu and gpu
- atomic tests should spin up a bunch of threads and slam the atomics and see if it breaks (we trust the compiler.. but do we really trust the compiler??)