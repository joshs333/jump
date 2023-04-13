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
    static_assert(kernel_interface<kernel_t>::has_index_kernel(), "kernel_t must have kernel(std::size_t) defined");

    if constexpr(kernel_interface<kernel_t>::has_index_kernel()) {
        for(std::size_t i = 0; i < count; ++i) {
            k.kernel(i);
        }
    }
}

template<typename kernel_t>
void iterate(const std::size_t& c1, const std::size_t& c2, const kernel_t& k) {
    static_assert(kernel_interface<kernel_t>::has_index_index_kernel(), "kernel_t must have kernel(std::size_t, std::size_t) defined");

    if constexpr(kernel_interface<kernel_t>::has_index_index_kernel()) {
        for(std::size_t i = 0; i < c1; ++i) {
            for(std::size_t j = 0; j < c2; ++j){
                k.kernel(i, j);
            }
        }
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

    void kernel(const std::size_t& idx) const {
        std::printf("%d\n", arr_.at(idx));
        arr_.at(idx) = 2;
    }

    jump::array<array_t> arr_;
}; /* struct access_kernel */

template<typename array_t>
struct access_kernel2 {
    access_kernel2(const jump::multi_array<array_t>& arr):
        arr_(arr)
    {}

    void kernel(const std::size_t& x1, const std::size_t& x2) const {
        std::printf("[%lu, %lu] %d\n", x1, x2, arr_.at(x1, x2));
    }

    jump::multi_array<array_t> arr_;
}; /* struct access_kernel2 */

int main(int argc, char** argv) {
    jump::array<int> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    jump::foreach(v, multiplier_kernel());
    jump::iterate(v.size(), access_kernel<int>(v));

    jump::multi_array<int> v2({10, 10, 10}, 10);
    jump::iterate(v2.shape(0), v2.shape(1), access_kernel2<int>(v2));
    // jump::iterate(v2.shape(), access_kernel<int>(v));


    return 0;
}

