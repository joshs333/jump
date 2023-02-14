/**
 * @file array_examples.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief some examples using the jump::array
 * @date 2022-07-26
 */
#include <jump/device_interface.hpp>
#include <jump/memory_buffer.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    std::printf("I got here.\n");
    // jump::memory_buffer buff;
    // buff.allocate(10, jump::memory_t::HOST);
    // buff.allocate<int>(10, jump::memory_t::UNIFIED);
    // buff.allocate<int>(10, jump::memory_t::DEVICE);

    // auto buf2 = buff.copy(jump::memory_t::HOST);

    return 0;
}



