#include <jump/array.hpp>

int main(int argc, char** argv) {
    jump::array<int> b1(10);
    jump::array<int> b2({1, 2, 3});
    jump::array<int> b3(10, 20);
    jump::array<int> b4(10, jump::array<int>::no_init{});

    return 0;
}
