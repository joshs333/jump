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

