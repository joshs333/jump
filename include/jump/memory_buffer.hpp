/**
 * @file memory_buffer.hpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief object containing a raw buffer of memory and controlling it's
 *  location in either host or device memory.
 * @version 0.1
 * @date 2022-10-05
 * @copyright Copyright (c) 2022 Joshua Spisak
 */
#ifndef JUMP_MEMORY_BUFFER_HPP_
#define JUMP_MEMORY_BUFFER_HPP_

#include <cstddef>
#include <cstdio>
#include <type_traits>

//! Just Multi-Processing namespace
namespace jump {

//! Types of memory that can be allocated / managed
enum class memory {
    //! Memory allocated to the host in the heap
    HOST,
    //! Memory allocated to a particular GPU device
    DEVICE,
    //! Memory allocated to unified memory, available to HOST and DEVICE
    UNIFIED
};

/**
 * @brief an object to own and manage a raw memory buffer
 */
class memory_buffer {
public:
    /**
     * @brief construct an empty memory buffer
     */
    memory_buffer()
    {}

    template<typename T = void>
    void allocate(std::size_t buffer_size) {
        std::size_t allocation_size;
        if constexpr(std::is_void<T>::value) {
            allocation_size = buffer_size;
        } else {
            allocation_size = buffer_size * sizeof(T);
        }

        std::printf("Allocate %lu\n", allocation_size);
    }

private:
    //! The primary location for this bfufer
    memory primary_location_ = memory::HOST;
    //! Pointer to a memory buffer (if allocated to host (or unified)
    void* host_data_ = nullptr;
    //! Pointer to a memory buffer (if allocated to device (or unified)
    void* device_data_ = nullptr;
    //! Size of the array (x dim)
    std::size_t size_ = 0;

}; /* class memory_buffer */


} /* namespace jump */

#endif /* JUMP_MEMORY_BUFFER_HPP_ */
