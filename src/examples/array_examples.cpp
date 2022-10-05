/**
 * @file shared_ptr_examples.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief some examples using the jump::shared_ptr
 * @date 2022-07-26
 */
// #include <jump/array.hpp>

#include <jump/memory_buffer.hpp>

int main(int argc, char** argv) {
    // jump::shared_ptr<int> b(new int{10});

    // jump::_array_helpers::indexing<3> indexing({10, 10, 10});

    jump::memory_buffer buff;
    buff.allocate(10);
    buff.allocate<int>(10);

    return 0;
}

