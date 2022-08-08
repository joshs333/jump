/**
 * @file shared_ptr_examples.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief some examples using the jump::shared_ptr
 * @date 2022-07-26
 */
#include <jump/shared_ptr.hpp>

int main(int argc, char** argv) {
    jump::shared_ptr<int> b(new int{10});

    return 0;
}

