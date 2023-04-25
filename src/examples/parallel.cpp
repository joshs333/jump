// JUMP
#include <jump/threadpool.hpp>
#include <jump/parallel.hpp>
#include <jump/multi_array.hpp>
#include <jump/array.hpp>

#include <mutex>

namespace jump {

/**
 * @brief arguments for parallel processing calls
 *  such as foreach and iterate
 */
struct par {
    enum target_t : uint8_t {
        seq = 0,
        threadpool = 1,
        cuda = 2
    };

    //! The execution target for the parallelized call
    target_t target = par::seq;
    //! If target == par::threadpool, the number of threads to use
    std::size_t thread_count = 3;

    //! If target == par::cuda, the number of threads per block in the cuda call
    std::size_t threads_per_block = 512;
};

//! Namespace for executors used by foreach or iterate calls to call threadpool
namespace parallel_executors {

template<typename kernel_t, typename array_t>
struct array_foreach {
    array_foreach(
        const kernel_t& kernel,
        const jump::array<array_t>& array
    ):
        kernel_(&kernel),
        array_(&array),
        index_(0)
    {}

    // bool control(jump::threadpool::context& context) const {
    //     std::scoped_lock l(index_mutex_);
    //     if(index_ < array_->size())
    //         return true;
    //     context.shutdown();
    //     return false;
    // }

    /// Shutdown can be called at any point during this execution
    void execute(jump::threadpool::context& context) const {
        if(context.shutdown_called()) return;
        std::size_t my_index = 0;

        while(true) {
            {
                std::scoped_lock l(index_mutex_);
                if(index_ >= array_->size()) {
                    context.shutdown();
                    return;
                }
                my_index = index_++;
            }

            kernel_->kernel(my_index, array_->at(my_index));
        }
    }

    const kernel_t* kernel_;
    const jump::array<array_t>* array_;
    mutable std::size_t index_;
    mutable std::mutex index_mutex_;
};

} /* namespace parallel_executors */

//! cuda kernels to enable foreach and iterate calls to use the cuda backend
namespace parallel_cuda_kernels {

template<typename kernel_t, typename array_t>
__global__
void array_foreach(
    kernel_t kernel,
    jump::array<array_t> array
) {
    std::size_t my_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(my_index >= array.size()) return;

    kernel.kernel(my_index, array.at(my_index));
}


}; /* namespace parallel_cuda_kernels */

template<typename kernel_t, typename array_t>
void foreach(
    jump::array<array_t>& array,
    kernel_t k,
    const par& options = par()
) {
    if(options.target == par::seq) {
        for(std::size_t i = 0; i < array.size(); ++i) {
            k.kernel(i, array.at(i));
        }
    } else if(options.target == par::threadpool) {
        // jump::threadpool pool;
        jump::threadpool().execute(parallel_executors::array_foreach(k, array), options.thread_count);
    } else if(options.target == par::cuda) {
        #if JUMP_ENABLE_CUDA
            if constexpr(kernel_interface<kernel_t>::device_compatible()) {
                // Setup
                if constexpr(kernel_interface<kernel_t>::to_device_defined()) {
                    k.to_device();
                }
                array.to_device();

                auto num_blocks = (int) std::ceil(array.size() / static_cast<float>(options.threads_per_block));

                // Call
                parallel_cuda_kernels::array_foreach<<<num_blocks, options.threads_per_block>>>(k, array);
                cudaDeviceSynchronize();

                // Cleanup!
                if constexpr(kernel_interface<kernel_t>::from_device_defined()) {
                    k.from_device();
                }
                array.from_device();
            } else {
                throw jump::device_incompatible_exception("unable to execute foreach call with cuda, kernel incompatible");
            }
        #else
            throw jump::no_cuda_exception("unable to parallelize with cuda");
        #endif
    } else {
        throw std::runtime_error("unknown target option");
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


template<std::size_t _dim_size, typename kernel_t>
void iterate(const jump::multi_indices<_dim_size>& shape, const kernel_t& k) {
    // if constexpr(kernel_interface<kernel_t>::has_index_index_kernel()) {
    //     for(std::size_t i = 0; i < c1; ++i) {
    //         for(std::size_t j = 0; j < c2; ++j){
    //             k.kernel(i, j);
    //         }
    //     }
    // }
}

template<typename kernel_t, typename array_t>
void foreach(jump::multi_array<array_t>& array, const std::size_t& iter_dim, const kernel_t& kernel) {
    auto shape = array.shape();

    // auto zero = 

}

}

struct multiplier_kernel {
    static const bool device_compatible = true;

    JUMP_INTEROPABLE
    void kernel(std::size_t& index, int& i) const {
        i = index * index;
        #if JUMP_ON_DEVICE
            for(int i = 0; i < 100; ++i) {
                auto b = i * i * i * i;
            }
        #else
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        #endif
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
        std::printf("[%lu, %lu] %d %lu\n", x1, x2, arr_.at(x1, x2), x1 * x2);
    }

    jump::multi_array<array_t> arr_;
}; /* struct access_kernel2 */

template<typename array_t>
struct access_kernel2shape {
    access_kernel2shape(const jump::multi_array<array_t>& arr):
        arr_(arr)
    {}

    void kernel(const jump::indices& idx) const {
        std::printf("[%lu, %lu] %d %lu\n", idx[0], idx[1], arr_.at(idx));
    }

    jump::multi_array<array_t> arr_;
}; /* struct access_kernel2shape */

template<typename array_t>
struct access_kernel3 {
    access_kernel3(const jump::multi_array<array_t>& arr):
        arr_(arr)
    {}

    void kernel(const std::size_t& x1, const std::size_t& x2, const std::size_t& x3) const {
        std::printf("[%lu, %lu] %d %lu\n", x1, x2, arr_.at(x1, x2), x1 * x2);
    }

    jump::multi_array<array_t> arr_;
}; /* struct access_kernel3 */


int main(int argc, char** argv) {
    // this should probably evolve into unit tests
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

    jump::array<int> v;

    for(int i = 0; i < 100; ++i) {
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


    bef = std::chrono::system_clock::now();
    jump::foreach(v, multiplier_kernel(), {.target = jump::par::cuda, .thread_count = 3});
    af = std::chrono::system_clock::now();
    double ns_cuda = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(af - bef).count());
    std::printf("%3.0f ms\n", ns_cuda);

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
    // jump::iterate(v.size(), access_kernel<int>(v));

    // std::printf("\n\n");

    // jump::multi_array<int> v2({10, 10, 10}, 10);
    // jump::iterate(v2.shape(0), v2.shape(1), access_kernel2<int>(v2));
    // jump::iterate(v2.shape(), access_kernel3<int>(v2));
    // jump::iterate(v2.shape(), access_kernel<int>(v));


    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::threadpool, .cores = {1, 2, 3}});
    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::cuda, .cores = {1, 2, 3}});
    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::seq, .cores = {1, 2, 3}});
    return 0;
}

