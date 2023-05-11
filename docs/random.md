# Random
We implemented some pseudo-random number generators with heavy guidance taken from Arvid Gerstmann's [blog post](https://arvid.io/2018/07/02/better-cxx-prng/).

I pulled in a box-muller projection for a normal distribution (with inspiration / implementation checks taken from the curand box-muller projection). Implementation of other distributions should be simpler.

The primary reason to implement it this way was to have a rng that would give the same results on GPU or CPU with the same seed. Alternatively
I could have written a joint wrapper around curand / std random utilities. Interestingly curand has both a host and device API, however they are not unified in the way I want. Additionally the host API seems to be designed around bulk random number generation offloaded to the GPU, which is not the functionality I am targetting (although jump can easily be used for this functionality).

Many thanks to the std library for interface inspiration and NVIDIA for making implementation details accesible for reference.
