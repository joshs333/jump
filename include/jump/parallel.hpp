/**
 * @file parallel.hpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief Various utilities for interopable parallel processing
 *  that can run sequentially or in parallel with a threadpool
 *  or cuda backend.
 * @date 2023-05-09
 */
#ifndef JUMP_PARALLEL_HPP_
#define JUMP_PARALLEL_HPP_

#include <jump/threadpool.hpp>
#include <jump/multi_array.hpp>
#include <jump/array.hpp>

#include <mutex>

namespace jump {

/**
 * @brief arguments for parallel processing calls
 *  such as foreach and iterate
 */
struct par {
    //! The possible targets to backend the parallel computation
    enum target_t : uint8_t {
        seq = 0,
        threadpool = 1,
        cuda = 2,
        unknown = 3
    };

    //! The execution target for the parallelized call
    target_t target = par::seq;
    //! If target == par::threadpool, the number of threads to use
    std::size_t thread_count = 3;

    //! If target == par::cuda, the number of threads per block in the cuda call
    //! TODO(jspisak): dynamically compute this better?
    std::size_t threads_per_block = 256;
    //! If target == par::cuda whether or not to perform to_device / from_device or trust the external caller to do it
    bool disable_device_transfers = false;
    // TODO(jspisak): device number (for cuda)

    //! Whether or not to print any debug calls (will be few and far between, this is more for developer use)
    bool debug = false;

    //! Builder design pattern - set debug
    par& set_debug(bool value = true) {
        debug = true;
        return *this;
    }

    //! Builder design pattern - set target
    par& set_target(target_t value) {
        target = value;
        return *this;
    }

    //! Builder design pattern - set disable_device_transfers
    par& set_disable_device_transfers(bool value = true) {
        disable_device_transfers = value;
        return *this;
    }

    //! Builder design pattern - set thread_count
    par& set_thread_count(std::size_t value) {
        thread_count = value;
        return *this;
    }

}; /* struct par */

//! The string representation of par::target_t
static const std::string_view par_target_t_strs[] = {
    "seq",
    "threadpool",
    "cuda",
    "unknown"
};

//! Namespace for executors used by foreach or iterate calls to call threadpool
namespace parallel_executors {

/**
 * @brief threadpool executor to execute a kernel in parallel over an array
 * @tparam kernel_t the kernel defining the function to execute
 * @tparam array_t the type containing in the array
 */
template<typename kernel_t, typename array_t>
struct array_foreach {
    /**
     * @brief create a new array_foreach executor
     * @param kernel the kernel to use
     * @param array the array to use
     * @note we assume that kernel and array will exist in the high-level caller
     *  for the lifetime of the executor, and only take a reference to it
     */
    array_foreach(
        const kernel_t& kernel,
        const jump::array<array_t>& array
    ):
        kernel_(&kernel),
        array_(&array),
        index_(0)
    {
        constexpr bool has_index_val_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
        constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
        static_assert(has_index_val_kernel || has_val_kernel, "kernel to an array foreach call must have function kernel(const std::size_t&, array_t&) or kernel(array_t&)");
    }

    // NOT USED, if array_.size() < thread_count, this could help
    //  reduce the overhead from spawning threads, we assume generally
    //  array_.size() is > thread_count and this would increase overhead
    // bool control(jump::threadpool::context& context) const {
    //     std::scoped_lock l(index_mutex_);
    //     if(index_ < array_->size())
    //         return true;
    //     context.shutdown();
    //     return false;
    // }

    /**
     * @brief executes the kernel in the executor
     * @param context the threadpool context we can use to signal
     *  shutdown once all elements have been processed
     */
    void execute(jump::threadpool::context& context) const {
        if(context.shutdown_called()) return;
        std::size_t my_index = 0;

        // process until all indices have been executed with
        while(true) {
            {
                std::scoped_lock l(index_mutex_);
                if(index_ >= array_->size()) {
                    context.shutdown();
                    return;
                }
                my_index = index_++;
            }


            constexpr bool has_index_val_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
            constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
            if constexpr(has_index_val_kernel)
                kernel_->kernel(my_index, array_->at(my_index));
            else
                kernel_->kernel(array_->at(my_index));
        }
    }

    //! const reference to the kernel (by pointer)
    const kernel_t* kernel_;
    //! const reference to the array (by pointer)
    const jump::array<array_t>* array_;
    //! The index that needs to be processed next (protected by mutex)
    mutable std::size_t index_;
    //! Mutex to protect the index_
    mutable std::mutex index_mutex_;

}; /* struct array_foreach */


/**
 * @brief threadpool executor to execute a kernel in parallel over an array
 * @tparam kernel_t the kernel defining the function to execute
 * @tparam array_t the type containing in the array
 */
template<typename kernel_t, typename array_t>
struct multi_array_foreach {
    /**
     * @brief create a new multi_array_foreach executor
     * @param kernel the kernel to use
     * @param array the array to use
     * @note we assume that kernel and array will exist in the high-level caller
     *  for the lifetime of the executor, and only take a reference to it
     */
    multi_array_foreach(
        const kernel_t& kernel,
        const jump::multi_array<array_t>& array
    ):
        kernel_(&kernel),
        array_(&array),
        index_(array.zero())
    {
        constexpr bool has_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
        constexpr bool has_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, array_t&>();
        constexpr bool has_i_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&, array_t&>();
        constexpr bool has_s_v_kernel = kernel_interface<kernel_t >::template has_kernel<const typename jump::multi_array<array_t>::indices&, array_t&>();
        constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
        constexpr bool has_kernel_defined = has_i_v_kernel | has_i_i_v_kernel 
            | has_i_i_i_v_kernel| has_s_v_kernel | has_val_kernel;
        static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");
    }

    /**
     * @brief executes the kernel in the executor
     * @param context the threadpool context we can use to signal
     *  shutdown once all elements have been processed
     */
    void execute(jump::threadpool::context& context) const {
        constexpr bool has_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
        constexpr bool has_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, array_t&>();
        constexpr bool has_i_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&, array_t&>();
        constexpr bool has_s_v_kernel = kernel_interface<kernel_t >::template has_kernel<const typename jump::multi_array<array_t>::indices&, array_t&>();
        constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
        constexpr bool has_kernel_defined = has_i_v_kernel | has_i_i_v_kernel 
            | has_i_i_i_v_kernel| has_s_v_kernel | has_val_kernel;
        static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");

        if(context.shutdown_called()) return;
        auto my_index = array_->zero();

        // process until all indices have been executed with
        while(true) {
            {
                std::scoped_lock l(index_mutex_);
                if(index_[0] >= array_->shape(0)) {
                    context.shutdown();
                    return;
                }
                my_index = index_;
                ++index_ %= array_->shape();
            }

            // we assume the checks happened at a higher-level, and don't throw from here
            if constexpr(has_val_kernel) {
                kernel_->kernel(array_->at(my_index));
            } else if constexpr(has_s_v_kernel) {
                kernel_->kernel(my_index, array_->at(my_index));
            } else {
                // then check for array dimension / compatible kernel
                if(array_->dims() == 1) {
                    if constexpr(has_i_v_kernel) {
                        kernel_->kernel(my_index[0], array_->at(my_index));
                    }
                } else if(array_->dims() == 2) {
                    if constexpr(has_i_i_v_kernel) {
                        kernel_->kernel(my_index[0], my_index[1], array_->at(my_index));
                    }
                } else if(array_->dims() == 3) {
                    if constexpr(has_i_i_i_v_kernel) {
                        kernel_->kernel(my_index[0], my_index[1], my_index[2], array_->at(my_index));
                    }
                }
            }
        }
    }

    //! const reference to the kernel (by pointer)
    const kernel_t* kernel_;
    //! const reference to the array (by pointer)
    const jump::multi_array<array_t>* array_;
    //! The index that needs to be processed next (protected by mutex)
    mutable typename jump::multi_array<array_t>::indices index_;
    //! Mutex to protect the index_
    mutable std::mutex index_mutex_;

}; /* struct multi_array_foreach */

/**
 * @brief threadpool executor to execute a kernel in parallel over an array
 * @tparam kernel_t the kernel defining the function to execute
 */
template<typename kernel_t>
struct iteration {
    /**
     * @brief create a new iteration executor
     * @param kernel the kernel to use
     * @param range the range to perform iteration over
     * @note we assume that kernel will exist in the high-level caller
     *  for the lifetime of the executor, and only take a reference to it
     */
    iteration(
        const kernel_t& kernel,
        const indices& range
    ):
        kernel_(&kernel),
        range_(range),
        index_(indices::zero(range.dims()))
    {
        constexpr bool has_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&>();
        constexpr bool has_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&>();
        constexpr bool has_i_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&>();
        constexpr bool has_s_kernel = kernel_interface<kernel_t >::template has_kernel<const jump::indices&>();
        constexpr bool has_kernel_defined = has_i_kernel || has_i_i_kernel || has_i_i_i_kernel || has_s_kernel;
        static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");
    }

    /**
     * @brief executes the kernel in the executor
     * @param context the threadpool context we can use to signal
     *  shutdown once all elements have been processed
     */
    void execute(jump::threadpool::context& context) const {
        constexpr bool has_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&>();
        constexpr bool has_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&>();
        constexpr bool has_i_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&>();
        constexpr bool has_s_kernel = kernel_interface<kernel_t >::template has_kernel<const jump::indices&>();
        constexpr bool has_kernel_defined = has_i_kernel || has_i_i_kernel || has_i_i_i_kernel || has_s_kernel;
        static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");

        if(context.shutdown_called()) return;
        auto my_index = index_;

        // process until all indices have been executed with
        while(true) {
            {
                std::scoped_lock l(index_mutex_);
                if(index_[0] >= range_[0]) {
                    context.shutdown();
                    return;
                }
                my_index = index_;
                ++index_ %= range_;
            }

            if constexpr(has_s_kernel) {
                kernel_->kernel(my_index);
            } else {
                if(range_.dims() == 1) {
                    if constexpr(has_i_kernel) {
                        kernel_->kernel(my_index[0]);
                    }
                }
                if(range_.dims() == 2) {
                    if constexpr(has_i_i_kernel) {
                        kernel_->kernel(my_index[0], my_index[1]);
                    }
                }
                if(range_.dims() == 3) {
                    if constexpr(has_i_i_i_kernel) {
                        kernel_->kernel(my_index[0], my_index[1], my_index[2]);
                    }
                }
            }
        }

    }

    //! const reference to the kernel (by pointer)
    const kernel_t* kernel_;
    //! The range to iterate over
    mutable indices range_;
    //! The index that needs to be processed next (protected by mutex)
    mutable indices index_;
    //! Mutex to protect the index_
    mutable std::mutex index_mutex_;

}; /* struct iteration */

} /* namespace parallel_executors */

//! cuda kernels to enable foreach and iterate calls to use the cuda backend
namespace parallel_cuda_kernels {

/**
 * @brief cuda kernel to execute a foreach command over an array
 * @tparam kernel_t the kernel type to use
 * @tparam array_t the type contained in the array
 * @param kernel the kernel to use (passed by value, kernel must ensure a bit-wise copy
 *  to the gpu will be good, ensure any pointers / arrays use jump types and to_device / from_device
 *  are properly defined).
 * @param array the array to parallelize over
 */
template<typename kernel_t, typename array_t>
__global__
void array_foreach(
    kernel_t kernel,
    jump::array<array_t> array
) {
    // recover the index
    std::size_t my_index = blockIdx.x * blockDim.x + threadIdx.x;

    // this can happen if the array size does not perfectly align with block size / warp thread count
    if(my_index >= array.size()) return;

    constexpr bool has_index_val_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
    constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
    static_assert(has_index_val_kernel || has_val_kernel, "kernel to an array foreach call must have function kernel(const std::size_t&, array_t&) or kernel(array_t&)");

    // execute!
    if constexpr(has_index_val_kernel)
        kernel.kernel(my_index, array.at(my_index));
    else if constexpr(has_val_kernel)
        kernel.kernel(array.at(my_index));
}

/**
 * @brief cuda kernel to execute a foreach command over an array
 * @tparam kernel_t the kernel type to use
 * @tparam array_t the type contained in the array
 * @param kernel the kernel to use (passed by value, kernel must ensure a bit-wise copy
 *  to the gpu will be good, ensure any pointers / arrays use jump types and to_device / from_device
 *  are properly defined).
 * @param array the array to parallelize over
 */
template<typename kernel_t, typename array_t>
__global__
void multi_array_foreach(
    kernel_t kernel,
    jump::multi_array<array_t> array
) {
    constexpr bool has_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
    constexpr bool has_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, array_t&>();
    constexpr bool has_i_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&, array_t&>();
    constexpr bool has_s_v_kernel = kernel_interface<kernel_t >::template has_kernel<const typename jump::multi_array<array_t>::indices&, array_t&>();
    constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
    constexpr bool has_kernel_defined = has_i_v_kernel | has_i_i_v_kernel 
        | has_i_i_i_v_kernel| has_s_v_kernel | has_val_kernel;
    static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");

    // recover the index
    std::size_t offset = blockIdx.x * blockDim.x + threadIdx.x;

    // this can happen if the array size does not perfectly align with block size / warp thread count
    if(offset >= array.size()) return;

    auto my_index = array.zero();
    my_index[array.dims() - 1] = offset;
    my_index %= array.shape();

    // we assume the checks happened at a higher-level, and don't throw from here
    if constexpr(has_val_kernel) {
        kernel.kernel(array.at(my_index));
    } else if constexpr(has_s_v_kernel) {
        kernel.kernel(my_index, array.at(my_index));
    } else {
        // then check for array dimension / compatible kernel
        if(array.dims() == 1) {
            if constexpr(has_i_v_kernel) {
                kernel.kernel(my_index[0], array.at(my_index));
            }
        } else if(array.dims() == 2) {
            if constexpr(has_i_i_v_kernel) {
                kernel.kernel(my_index[0], my_index[1], array.at(my_index));
            }
        } else if(array.dims() == 3) {
            if constexpr(has_i_i_i_v_kernel) {
                kernel.kernel(my_index[0], my_index[1], my_index[2], array.at(my_index));
            }
        }
    }
}

/**
 * @brief iterate a kernel over a range
 * @tparam kernel_t the kernel to use
 * @param options the parallel options struct for this call (added for debug purposes, can likely be removed)
 * @param kernel the kernel encoding the function to parallelize over
 * @param range the range to iterate over
 */
template<typename kernel_t>
__global__
void iteration(
    par options,
    kernel_t kernel,
    indices range
) {
    constexpr bool has_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&>();
    constexpr bool has_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&>();
    constexpr bool has_i_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&>();
    constexpr bool has_s_kernel = kernel_interface<kernel_t >::template has_kernel<const jump::indices&>();
    constexpr bool has_kernel_defined = has_i_kernel || has_i_i_kernel || has_i_i_i_kernel || has_s_kernel;
    static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");

    // auto range = rangess;

    // recover the index
    std::size_t offset = blockIdx.x * blockDim.x + threadIdx.x;

    // this can happen if the array size does not perfectly align with block size / warp thread count
    if(offset >= range.offset()) return;

    auto my_index = indices::zero(range.dims());
    my_index[range.dims() - 1] = offset;

    // TODO(jspisak): fix bug??
    //  when I use this, in some cases it produces a bug where the result is not correct
    //  then if I remove code in the kernel it fixes the bug.. adding printout statements
    //  in the operator%= implementation in multi_indices it also fixes it.. much confusing
    //  honestly I think it's a compiler bug.
    // my_index = my_index % range;

    //  This was another 'fix'
    // if(range.dims() > 1)
    //     my_index %= range

    my_index = my_index.operator%(range);

    if(my_index[0] >= range[0])
        return;


    if constexpr(has_s_kernel) {
        kernel.kernel(my_index);
    } else {
        if(my_index.dims() == 1) {
            if constexpr(has_i_kernel) {
                kernel.kernel(my_index[0]);
            }
        }
        if(my_index.dims() == 2) {
            if constexpr(has_i_i_kernel) {
                kernel.kernel(my_index[0], my_index[1]);
            }
        }
        if(my_index.dims() == 3) {
            if constexpr(has_i_i_i_kernel) {
                kernel.kernel(my_index[0], my_index[1], my_index[2]);
            }
        }
    }
}

}; /* namespace parallel_cuda_kernels */

/**
 * @brief executes a function (kernel) over each element in an array
 * @tparam kernel_t the kernel that defines the parallelized function
 * @tparam array_t the type in the array we are parallelizing over
 * @param array the array holding data we are executing soemthing foreach
 * @param kernel the kernel defining the parallelized function
 * @param options options for execution
 */
template<typename kernel_t, typename array_t>
void foreach(
    jump::array<array_t>& array,
    kernel_t kernel,
    const par& options = par()
) {
    constexpr bool has_index_val_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
    constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
    static_assert(has_index_val_kernel || has_val_kernel, "kernel to an array foreach call must have function kernel(const std::size_t&, array_t&) or kernel(array_t&)");

    if constexpr(has_index_val_kernel || has_val_kernel) {
        // sequential is just a for loop
        if(options.target == par::seq) {
            for(std::size_t i = 0; i < array.size(); ++i) {
                if constexpr(has_index_val_kernel)
                    kernel.kernel(i, array.at(i));
                else if constexpr(has_val_kernel)
                    kernel.kernel(array.at(i));
            }
        
        // threadpool executes the kernel using a threadpool
        } else if(options.target == par::threadpool) {
            jump::threadpool().execute(parallel_executors::array_foreach(kernel, array), options.thread_count);
        
        // the cuda option executes in parallel on GPU
        } else if(options.target == par::cuda) {
            #if JUMP_ENABLE_CUDA
                if constexpr(kernel_interface<kernel_t>::device_compatible()) {
                    // Setup
                    if constexpr(kernel_interface<kernel_t>::to_device_defined()) {
                        if(!options.disable_device_transfers)
                            kernel.to_device();
                    }
                    if(!options.disable_device_transfers)
                        array.to_device();

                    auto num_blocks = (int) std::ceil(array.size() / static_cast<float>(options.threads_per_block));

                    // Call
                    parallel_cuda_kernels::array_foreach<<<num_blocks, options.threads_per_block>>>(kernel, array);
                    auto error = cudaDeviceSynchronize();
                    if(error != cudaSuccess)
                        throw jump::cuda_error_exception(error);

                    // Cleanup!
                    if constexpr(kernel_interface<kernel_t>::from_device_defined()) {
                        if(!options.disable_device_transfers)
                            kernel.from_device();
                    }
                    if(!options.disable_device_transfers)
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
}

/**
 * @brief executes a function (kernel) over each element in an multi_array
 * @tparam kernel_t the kernel that defines the parallelized function
 * @tparam array_t the type in the multi_array we are parallelizing over
 * @param array the multi_array holding data we are executing soemthing foreach
 * @param kernel the kernel defining the parallelized function
 * @param options options for execution
 */
template<typename kernel_t, typename array_t>
void foreach(
    jump::multi_array<array_t>& array,
    kernel_t kernel,
    const par& options = par()
) {
    constexpr bool has_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, array_t&>();
    constexpr bool has_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, array_t&>();
    constexpr bool has_i_i_i_v_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&, array_t&>();
    constexpr bool has_s_v_kernel = kernel_interface<kernel_t >::template has_kernel<const typename jump::multi_array<array_t>::indices&, array_t&>();
    constexpr bool has_val_kernel = kernel_interface<kernel_t >::template has_kernel<array_t&>();
    constexpr bool has_kernel_defined = has_i_v_kernel | has_i_i_v_kernel 
        | has_i_i_i_v_kernel| has_s_v_kernel | has_val_kernel;
    static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");

    if constexpr(!has_val_kernel && !has_s_v_kernel) {
        // then check for array dimension / compatible kernel
        if(array.dims() == 1) {
            if constexpr(!has_i_v_kernel) {
                throw std::runtime_error("kernel has dims of 1, but no kernel() function for it");
            }
        } else if(array.dims() == 2) {
            if constexpr(!has_i_i_v_kernel) {
                throw std::runtime_error("kernel has dims of 2, but no kernel() function for it");
            }
        } else if(array.dims() == 3) {
            if constexpr(!has_i_i_i_v_kernel) {
                throw std::runtime_error("kernel has dims of 3, but no kernel() function for it");
            }
        } else {
            // we shouldn't get here unless running with a very high-dimensional multi_array
            throw std::runtime_error("unable to handle dimensionality of " + std::to_string(array.dims()));
        }
    }


    if constexpr(has_kernel_defined) {
        // sequential is just a for loop
        if(options.target == par::seq) {
            for(auto i = array.zero(); i[0] < array.shape(0); ++i %= array.shape()) {
                auto s = array.shape();
                // first check if we have a non-dimensionality dependant kernel function
                if constexpr(has_val_kernel) {
                    kernel.kernel(array.at(i));
                } else if constexpr(has_s_v_kernel) {
                    kernel.kernel(i, array.at(i));
                } else {
                    // then check for array dimension / compatible kernel
                    if(array.dims() == 1) {
                        if constexpr(has_i_v_kernel) {
                            kernel.kernel(i[0], array.at(i));
                        }
                    } else if(array.dims() == 2) {
                        if constexpr(has_i_i_v_kernel) {
                            kernel.kernel(i[0], i[1], array.at(i));
                        }
                    } else if(array.dims() == 3) {
                        if constexpr(has_i_i_i_v_kernel) {
                            kernel.kernel(i[0], i[1], i[2], array.at(i));
                        }
                    }
                }
            }
        
        // threadpool executes the kernel using a threadpool
        } else if(options.target == par::threadpool) {
            jump::threadpool().execute(parallel_executors::multi_array_foreach(kernel, array), options.thread_count);
        
        // the cuda option executes in parallel on GPU
        } else if(options.target == par::cuda) {
            #if JUMP_ENABLE_CUDA
                if constexpr(kernel_interface<kernel_t>::device_compatible()) {
                    // Setup
                    if constexpr(kernel_interface<kernel_t>::to_device_defined()) {
                        if(!options.disable_device_transfers)
                            kernel.to_device();
                    }
                    if(!options.disable_device_transfers)
                        array.to_device();

                    // Call TODO(jspisak): ensure that this method of handling blocks / threads doesn't limit size
                    auto num_blocks = (int) std::ceil(array.size() / static_cast<float>(options.threads_per_block));
                    parallel_cuda_kernels::multi_array_foreach<<<num_blocks, options.threads_per_block>>>(kernel, array);
                    auto error = cudaDeviceSynchronize();
                    if(error != cudaSuccess)
                        throw jump::cuda_error_exception(error);

                    // Cleanup!
                    if constexpr(kernel_interface<kernel_t>::from_device_defined()) {
                        if(!options.disable_device_transfers)
                            kernel.from_device();
                    }
                    if(!options.disable_device_transfers)
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
}

/**
 * @brief parallelizes a call to iterate over a range
 * @tparam kernel_t the struct encoding kernel functions to use
 * @param range the range to iterate over
 * @param kernel the kernel to use
 * @param options the options to use for parallelizing
 */
template<typename kernel_t>
void iterate(
    const indices& range,
    kernel_t kernel,
    const par& options = par()
) {
    constexpr bool has_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&>();
    constexpr bool has_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&>();
    constexpr bool has_i_i_i_kernel = kernel_interface<kernel_t >::template has_kernel<const std::size_t&, const std::size_t&, const std::size_t&>();
    constexpr bool has_s_kernel = kernel_interface<kernel_t >::template has_kernel<const jump::indices&>();
    constexpr bool has_kernel_defined = has_i_kernel || has_i_i_kernel || has_i_i_i_kernel || has_s_kernel;
    static_assert(has_kernel_defined, "kernel_t does not have any proper kernel functions defined.");

    if constexpr(!has_s_kernel) {
        if(range.dims() == 1) {
            if constexpr(!has_i_kernel) {
                throw std::runtime_error("kernel has no call for a 1d range");
            }
        }
        if(range.dims() == 2) {
            if constexpr(!has_i_i_kernel) {
                throw std::runtime_error("kernel has no call for a 2d range");
            }
        }
        if(range.dims() == 3) {
            if constexpr(!has_i_i_i_kernel) {
                throw std::runtime_error("kernel has no call for a 3d range");
            }
        }
    }

    if constexpr(has_kernel_defined) {
        if(options.target == par::seq) {
            for(auto i = indices::zero(range.dims()); i[0] < range[0]; ++i %= range) {
                if constexpr(has_s_kernel) {
                    kernel.kernel(i);
                } else {
                    if(range.dims() == 1) {
                        if constexpr(has_i_kernel) {
                            kernel.kernel(i[0]);
                        }
                    }
                    if(range.dims() == 2) {
                        if constexpr(has_i_i_kernel) {
                            kernel.kernel(i[0], i[1]);
                        }
                    }
                    if(range.dims() == 3) {
                        if constexpr(has_i_i_i_kernel) {
                            kernel.kernel(i[0], i[1], i[2]);
                        }
                    }
                }
            }
        } else if(options.target == par::threadpool) {
            jump::threadpool().execute(parallel_executors::iteration(kernel, range), options.thread_count);
        } else if(options.target == par::cuda) {
            #if JUMP_ENABLE_CUDA
                if constexpr(kernel_interface<kernel_t>::device_compatible()) {
                    // Setup
                    if constexpr(kernel_interface<kernel_t>::to_device_defined()) {
                        if(!options.disable_device_transfers)
                            kernel.to_device();
                    }

                    // Call TODO(jspisak): ensure that this method of handling blocks / threads doesn't limit size
                    auto num_blocks = (int) std::ceil(range.offset() / static_cast<float>(options.threads_per_block));
                    parallel_cuda_kernels::iteration<<<num_blocks, options.threads_per_block>>>(options, kernel, range);
                    auto error = cudaDeviceSynchronize();
                    if(error != cudaSuccess)
                        throw jump::cuda_error_exception(error);

                    // Cleanup!
                    if constexpr(kernel_interface<kernel_t>::from_device_defined()) {
                        if(!options.disable_device_transfers)
                            kernel.from_device();
                    }
                } else {
                    throw jump::device_incompatible_exception("unable to execute iterate call with cuda, kernel incompatible");
                }
            #else
                throw jump::no_cuda_exception("unable to parallelize with cuda");
            #endif
        } else {
            throw std::runtime_error("Somehow got here?? unknown target.");
        }
    }
}

/**
 * @brief parallelizes a call to iterate over a 1d range
 * @tparam kernel_t the struct encoding kernel functions to use
 * @param r1 the size of the range to iterate over
 * @param kernel the kernel to use
 * @param options the options to use for parallelizing
 */
template<typename kernel_t>
void iterate(
    const std::size_t& r1,
    kernel_t kernel,
    const par& options = par()
) {
    iterate(indices(r1), kernel, options);
}

/**
 * @brief parallelizes a call to iterate over a 1d range
 * @tparam kernel_t the struct encoding kernel functions to use
 * @param r1 the size of the range to iterate over (dim 1)
 * @param r2 the size of the range to iterate over (dim 2)
 * @param kernel the kernel to use
 * @param options the options to use for parallelizing
 */
template<typename kernel_t>
void iterate(
    const std::size_t& r1,
    const std::size_t& r2,
    kernel_t kernel, 
    const par& options = par()
) {
    iterate(indices(r1, r2), kernel, options);
}


/**
 * @brief parallelizes a call to iterate over a 1d range
 * @tparam kernel_t the struct encoding kernel functions to use
 * @param r1 the size of the range to iterate over (dim 1)
 * @param r2 the size of the range to iterate over (dim 2)
 * @param r3 the size of the range to iterate over (dim 3)
 * @param kernel the kernel to use
 * @param options the options to use for parallelizing
 */
template<typename kernel_t>
void iterate(
    const std::size_t& r1,
    const std::size_t& r2,
    const std::size_t& r3,
    kernel_t kernel,
    const par& options = par()
) {
    iterate(indices(r1, r2, r3), kernel, options);
}

} /* namepace jump*/

#endif /* JUMP_PARALLEL_HPP_ */
