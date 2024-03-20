// JUMP
#include <jump/multi_array.hpp>

// STD
#include <cstdio>
#include <iostream>

using namespace std;

__global__
void r(jump::multi_array<int> t) {
    t.at(4, 5, 2);
}


int main(int argc, char** argv) {
    std::printf("At beginning\n");

    jump::multi_array<int> a({5, 5}, jump::memory_t::UNIFIED);
    a.at(0, 0) = 1;
    std::printf("%d\n", a.at(0, 0));
    // a.to_device();


    // std::string bob;
    // auto k = static_cast<std::size_t>(bob);

    // r<<<1, 1>>>(a);
    // cudaDeviceSynchronize();

    return 0;
}
