
#include <jpc/test.h>

__global__
void stuff() {
}


void doStuff() {
    auto t = std::chrono::system_clock::now();

    stuff<<<1,1>>>();
}