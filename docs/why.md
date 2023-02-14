# Why
Why write yet another parallel computing library. There are two key features that I wanted but was unable to find in other parallelization libraries.
- clean interopable data structures
- run-time parallelization mechanism selection

For example [hemi](https://github.com/harrism/hemi) has an array class that manages cuda memory in a friendly way, but the array is still turned into a raw pointer come execution time. The nvidia parallel algorithms library [thrust](https://github.com/NVIDIA/thrust) does better, it has device and host vectors that provide a high-level interface with iterators which stl users might find friendly. But in neither of these libraries is there a clean way to see if CUDA is available at runtime and flip a switch to change the parallelization backend and memory model. While thrust provides a few parallelization backends, they can only be switched between at compile-time.

The goal of this library is to provide a more flexible run-time options, perhaps at the cost of a slightly more complicated interface (eg: priorize lambdas that can encode constepr info on whether the functor is interopable or not*). This in turn should make code more cross-platform (eg: build with cuda enabled, package for computers without CUDA that can still parallelize on CPU).

TODO(jspisak): make a separate readme detailing this after I figure it out
\* what do I mean constepr info on whether it's interopable or not?
