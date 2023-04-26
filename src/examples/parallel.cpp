// JUMP
#include <jump/parallel.hpp>

// STD
#include <cstdio>


struct multiplier_kernel {
    static const bool device_compatible = true;

    JUMP_INTEROPABLE
    void kernel(const std::size_t& idxa, const std::size_t& idxb, const std::size_t& idxc, int& i) const {
        std::printf(":) %lu %lu %lu ... %d\n", idxa, idxb, idxc, i);
        i = (idxa + 1) * (idxb + 1) * (idxc + 1);
    }

    JUMP_INTEROPABLE
    void kernel(const std::size_t& index, int& i) const {
        i = index * index;
        std::printf("Functor.. %lu %d\n", index, i);
    }
}; /* struct multiplier_kernel */

void index_testing() {
    jump::indices idx(5, 3, 3);
    jump::indices idx2(5, 3, 4);
    jump::indices shh(6, 4, 7);
    std::printf("%lu, %lu, %lu\n", shh[0], shh[1], shh[2]);
    std::printf("%lu, %lu, %lu <=> %lu %lu %lu ==(%s) <(%s) <=(%s) >(%s) >=(%s) !=(%s)\n", idx[0], idx[1], idx[2], idx2[0], idx2[1], idx2[2], 
        idx == idx2 ? "true" : "false", idx < idx2 ? "true" : "false", idx <= idx2 ? "true" : "false",
        idx > idx2 ? "true" : "false", idx >= idx2 ? "true" : "false", idx != idx2 ? "true" : "false"
    );
    ++idx %= shh;
    std::printf("%lu, %lu, %lu <=> %lu %lu %lu ==(%s) <(%s) <=(%s) >(%s) >=(%s) !=(%s)\n", idx[0], idx[1], idx[2], idx2[0], idx2[1], idx2[2],
        idx == idx2 ? "true" : "false", idx < idx2 ? "true" : "false", idx <= idx2 ? "true" : "false",
        idx > idx2 ? "true" : "false", idx >= idx2 ? "true" : "false", idx != idx2 ? "true" : "false"
    );
    ++idx %= shh;
    std::printf("%lu, %lu, %lu <=> %lu %lu %lu ==(%s) <(%s) <=(%s) >(%s) >=(%s) !=(%s)\n", idx[0], idx[1], idx[2], idx2[0], idx2[1], idx2[2],
        idx == idx2 ? "true" : "false", idx < idx2 ? "true" : "false", idx <= idx2 ? "true" : "false",
        idx > idx2 ? "true" : "false", idx >= idx2 ? "true" : "false", idx != idx2 ? "true" : "false"
    );
    ++idx %= shh;
    std::printf("%lu, %lu, %lu\n", idx[0], idx[1], idx[2]);
    ++idx %= shh;
    std::printf("%lu, %lu, %lu\n", idx[0], idx[1], idx[2]);
    std::printf("\n\n");
}

void array_foreach_testing() {
    jump::array<int> v;
    for(int i = 0; i < 2; ++i) {
        v.push_back(i);
    }
    auto bef = std::chrono::system_clock::now();
    jump::foreach(v, multiplier_kernel());
    auto af = std::chrono::system_clock::now();
    double ns_seq = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(af - bef).count());
    std::printf("%3.0f ms\n", ns_seq);

    bef = std::chrono::system_clock::now();
    jump::foreach(v, multiplier_kernel(), {.target = jump::par::threadpool, .thread_count = 3});
    af = std::chrono::system_clock::now();
    double ns_threadpool = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(af - bef).count());
    std::printf("%3.0f ms\n", ns_threadpool);

    double agg = 0;
    int count = 0;
    double ns_cuda = 0;
    for(int i = 0; i < 1; ++i) {
        bef = std::chrono::system_clock::now();
        jump::foreach(v, multiplier_kernel(), {.target = jump::par::cuda, .thread_count = 3});
        af = std::chrono::system_clock::now();
        ns_cuda = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(af - bef).count());
        std::printf("%03d: Got %3.0f ms\n", i, ns_cuda);
        if(i > 2) {
            agg += ns_cuda;
            count++;
        }
    }
    std::printf("%3.0f ms\n", agg / count);
    // ns_cuda = (agg / count) * 1e-6;

    std::printf("Threadpool took %3.0f ms %s than seq\n",
        std::abs(ns_threadpool - ns_seq),
        ns_threadpool > ns_seq ? "more" : "less"
    );

    std::printf("Threadpool took %3.0f ms %s than cuda\n",
        std::abs(ns_threadpool - ns_cuda),
        ns_threadpool > ns_cuda ? "more" : "less"
    );

    std::printf("multiplier_kernel device compatible?? %s\n",
        jump::kernel_interface<multiplier_kernel>::device_compatible() ? "true" : "false"
    );
}

void multi_array_foreach_testing() {
    jump::multi_array<int> v2({2, 2, 2}, 10);
    jump::foreach(v2, multiplier_kernel(), {.target = jump::par::seq});
    std::printf("TP\n");
    jump::foreach(v2, multiplier_kernel(), {.target = jump::par::threadpool});
    std::printf("CU\n");
    jump::foreach(v2, multiplier_kernel(), {.target = jump::par::cuda});
}

struct iterate_indice_kernel {
    JUMP_INTEROPABLE
    void kernel(const jump::indices& idx) const {
        if(idx.dims() == 1) {
            std::printf("1d: %lu\n", idx[0]);
        }
        if(idx.dims() == 2) {
            std::printf("2d: %lu %lu\n", idx[0], idx[1]);
        }
        if(idx.dims() == 3) {
            std::printf("3d: %lu %lu %lu\n", idx[0], idx[1], idx[2]);
        }
    }
};

struct iterate_1d_kernel {
    JUMP_INTEROPABLE
    void kernel(const std::size_t& idx) const {
        std::printf("fixed 1d %lu\n", idx);
    }
};

struct iterate_2d_kernel {
    JUMP_INTEROPABLE
    void kernel(const std::size_t& idxa, const std::size_t& idxb) const {
        std::printf("fixed 2d %lu %lu\n", idxa, idxb);
    }
};

struct iterate_3d_kernel {
    JUMP_INTEROPABLE
    void kernel(const std::size_t& idxa, const std::size_t& idxb, const std::size_t& idxc) const {
        std::printf("fixed 3d %lu %lu %lu\n", idxa, idxb, idxc);
    }
};

void iterate_testing() {
    jump::iterate(jump::indices(3, 3), iterate_indice_kernel(), {.target = jump::par::seq});
    jump::iterate(jump::indices(3, 3), iterate_2d_kernel(), {.target = jump::par::threadpool});
    jump::iterate(jump::indices(3, 3), iterate_indice_kernel(), {.target = jump::par::cuda});
    jump::iterate(3, iterate_indice_kernel(), {.target = jump::par::threadpool});
    jump::iterate(3, 3, iterate_indice_kernel(), {.target = jump::par::threadpool});
    jump::iterate(3, 3, 3, iterate_indice_kernel(), {.target = jump::par::threadpool});
}

struct int_print_kernel {
    int_print_kernel(const jump::multi_array<int>& arr):
        arr_(arr)
    {}

    JUMP_INTEROPABLE
    void kernel(const std::size_t& x1, const std::size_t& x2, const std::size_t& x3) const {
        std::printf("[%lu, %lu, %lu] %d\n", x1, x2, x3, arr_.at(x1, x2, x3));
    }

    JUMP_INTEROPABLE
    void kernel(const std::size_t& x1, const std::size_t& x2) const {
        std::printf("[%lu, %lu] %d\n", x1, x2, arr_.at(x1, x2));
    }

    JUMP_INTEROPABLE
    void kernel(const std::size_t& x1) const {
        std::printf("[%lu] %d\n", x1, arr_.at(x1));
    }

    void to_device() {
        arr_.to_device();
    }

    void from_device() {
        arr_.from_device();
    }

    jump::multi_array<int> arr_;
}; /* struct access_kernel3 */

void iterate_multi_array_testing() {
    jump::multi_array<int> arr1({9}, 1);
    jump::multi_array<int> arr2({3, 3}, 2);
    jump::multi_array<int> arr3({2, 2, 2}, 3);
    jump::iterate(arr1.shape(), int_print_kernel(arr1), {.target = jump::par::seq});
    jump::iterate(arr2.shape(), int_print_kernel(arr2), {.target = jump::par::threadpool});
    jump::iterate(arr3.shape(), int_print_kernel(arr3), {.target = jump::par::cuda});
}


int main(int argc, char** argv) {
    // this should probably evolve into unit tests
    index_testing();
    array_foreach_testing();
    multi_array_foreach_testing();
    iterate_testing();
    iterate_multi_array_testing();

    // std::printf("\n\n");

    // jump::iterate(v2.shape(0), v2.shape(1), access_kernel2<int>(v2));
    // jump::iterate(v2.shape(), access_kernel3<int>(v2));
    // jump::iterate(v2.shape(), access_kernel<int>(v));


    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::threadpool, .cores = {1, 2, 3}});
    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::cuda, .cores = {1, 2, 3}});
    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::seq, .cores = {1, 2, 3}});
    return 0;
}

