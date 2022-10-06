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
    if constexpr(jump::on_device()) {
        deviceOnlyFunc();
        std::printf("Hello you! Device: %s\n", jump::on_device() ? "true" : "false");
    } else {
        std::printf("Hello you! Host: %s\n", jump::on_device() ? "true" : "false");
    }
}

__global__
void dummyKernel() {
    testPrint();
    jump::device_thread_sync();
    // __syncthreads();
}

int main(int argc, char** argv) {
    // jump::shared_ptr<int> b(new int{10});

    // jump::_array_helpers::indexing<3> indexing({10, 10, 10});

    testPrint();
    jump::memory_buffer buff;
    buff.allocate(10);
    buff.allocate<int>(10);


    return 0;
}

