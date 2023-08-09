// JUMP
#include <jump/parallel.hpp>
#include <jump/array.hpp>

// STD
#include <cstdio>

namespace jump {

namespace reduction_helpers {
}

template<typename T>
struct min {
    static T kernel(const T& a, const T& b) {
        if(a < b)
            return a;
        else
            return b;
    }
};

template<typename array_t, typename kernel_t>
array_t reduce(const jump::array<array_t>& arr, const kernel_t& k, const jump::par& options = jump::par()) {
    if(arr.size() < 2)
        throw std::runtime_error("array size (" + std::to_string(arr.size()) + ") must be greater than 1 to reduce.");

    if(options.target == jump::par::seq) {
        auto v = k.kernel(arr.at(0), arr.at(1));
        for(std::size_t i = 2; i < arr.size(); ++i) {
            v = k.kernel(v, arr.at(i));
        }
        return v;
    } else if(options.target == jump::par::cuda) {
    
    } else {
        std::printf("NOT IMPLEMENTED!\n");
    }
    return arr[0];
}

} /* namespace jump */

int main(int argc, char** argv) {
    std::printf("Hello you\n");

    jump::array<int> a(10);

    for(std::size_t i = 0; i < a.size(); ++i) {
        a.at(i) = a.size() - i;
        a.at(i) = a.size() - i - 1;
        std::printf("arr[%lu] = %d\n", i, a.at(i));
    }

    auto k = jump::reduce(a, jump::min<int>(), jump::par().set_target(jump::par::seq));
    std::printf("Min is: %d\n", k);

    return 0;
}
