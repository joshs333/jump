/**
 * @file array_examples.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief some examples using the jump::array
 * @date 2022-07-26
 */
#include <jump/device_interface.hpp>
#include <jump/memory_buffer.hpp>
#include <cstdio>

JUMP_DEVICE_ONLY
void deviceOnlyFunc() {
    std::printf("I am device only!!!\n");
}

JUMP_INTEROPABLE
void testPrint() {
    #if JUMP_ON_DEVICE
        deviceOnlyFunc();
        std::printf("Hello you! Device: %s\n", "true");
    #else
        std::printf("Hello yous! Host: %s\n", "false");
    #endif
    jump::device_thread_sync();
}

__global__
void dummyKernel() {
    std::printf("Heya you\n");
    testPrint();
    jump::device_thread_sync();
}

int main(int argc, char** argv) {
    // jump::shared_ptr<int> b(new int{10});

    // jump::_array_helpers::indexing<3> indexing({10, 10, 10});

    testPrint();
    jump::memory_buffer buff;
    buff.allocate(10);
    buff.allocate<int>(10);

    dummyKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}

