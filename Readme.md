# High-Level Design
Interopable memory buffers.

Multi-size arrays.

Parallel execution with GPU or thread pools.

auto par = jump::threadpool()

jump::memory_buffer buf("heya :)");
buf.allocate<int>(1000);

auto buf2 = buf.copy();

jump::foreach(jump::array({10, 10, 10}), jump::indices({0, 1}), );

The memory buffer is a form of shared pointer where references to the buffer are tracked.
Unless an explicit copy call is executed, the buffer is shared between all arrays / buffers viewing it.

An array by default will create a new memory buffer, unless it is constructed with a passed in memory buffer.

octree 
- point based
- container based (ray trace to first intersection, all intersections)

## Open Questions / To-Do's
- should I have made the kernels / executors const to be passed to the iterate / foreach calls (same for threadpool executor interfaces)
- how to handle different (max) dimension indices / multi_arrays
- indices operators (and handling the difference between representing size vs an index)
- make foreach / iterate callable from device??
- make cuda calls for multi_array use 3d blocks or whatevs? (easy to hit limit with three dimensional array)
- flesh out unit tests for parallel, threadpool, new multi_array features with the indices
- make array / multi_array to not have shared_ptr behavior (change the copy / assignment operators to clone the memory buffer or smthn)

# Foreach kernels
struct k {
    void kernel()
}
