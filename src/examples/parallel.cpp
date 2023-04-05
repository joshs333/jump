// JUMP
#include <jump/parallel.hpp>
#include <jump/multi_array.hpp>
#include <jump/array.hpp>

namespace jump {

template<typename kernel, typename array_t>
void foreach(jump::array<array_t>& array, const kernel& k) {
    for(std::size_t i = 0; i < array.size(); ++i) {
        k(i, array.at(i));
    }
}

template<typename kernel_t>
void iterate(const std::size_t& count, const kernel_t& k) {
    for(int i = 0; i < count; ++i) {
        k(i);
    }
}

template<typename kernel_t, typename array_t>
void foreach(jump::multi_array<array_t>& array, const std::size_t& iter_dim, const kernel_t& kernel) {
    auto shape = array.shape();

    // auto zero = 

}

}

struct multiplier_kernel {
    void operator()(std::size_t& index, int& i) const {
        i = index * index;
        std::printf("Functor.. %lu %d\n", index, i);
    }
}; /* struct multiplier_kernel */

template<typename array_t>
struct access_kernel {
    access_kernel(const jump::array<array_t>& arr):
        arr_(arr)
    {}

    void operator()(const int& idx) const {
        arr_.at(idx) = 2;
        std::printf("%d\n", arr_.at(idx));
    }

    jump::array<array_t> arr_;
}; /* struct access_kernel */

template<typename array_t>
struct print_kernel {

    void operator()(array_t& v) {
        std::printf("%s\n", std::to_string(v).c_str());
    }

};

int main(int argc, char** argv) {
    jump::array<int> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    jump::foreach(v, multiplier_kernel());
    jump::iterate(v.size(), access_kernel<int>(v));

    jump::multi_array<int> v2({10, 10, 10}, 10);


    // jump::foreach(v2, print_kernel<int>());

    // jump::iterate(v2.shape(), )

    return 0;
}

