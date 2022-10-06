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

// STD
#include <cstddef>
#include <stdexcept>
#include <cstdio>
#include <type_traits>

// APPLICATION
#include <jump/device_interface.hpp>

//! Just Multi-Processing namespace
namespace jump {

//! Types of memory that can be allocated / managed
enum class memory_t {
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
    T* data() {
        #if JUMP_ON_DEVICE
            return reinterpret_cast<T*>(device_data_);
        #else
            return reinterpret_cast<T*>(host_data_);
        #endif
    }

    template<typename T = void>
    T* host_data() {
        return reinterpret_cast<T*>(host_data_);
    }

    template<typename T = void>
    T* device_data() {
        return reinterpret_cast<T*>(device_data_);
    }

    /**
     * @brief allocates a section of memory at some memory location
     * @tparam T the type to allocate (default is void)
     * @param buffer_size size of the buffer to allocate (automatic * sizeof(T) if T != void)
     * @param location the memory location to allocate in (HOST, DEVICE, or UNIFIED)
     */
    template<typename T = void>
    void allocate(std::size_t buffer_size, memory_t location = memory_t::HOST) {
        // Deallocate if we already have data
        deallocate();
        
        // We refuse to allocate nothing
        if(buffer_size == 0) {
            return;
        }
    
        // Compute the allocation size based on the type we are allocating
        std::size_t allocation_size;
        if constexpr(std::is_void<T>::value) {
            allocation_size = buffer_size;
        } else {
            allocation_size = buffer_size * sizeof(T);
        }

        primary_location_ = location;
        if(location == memory_t::HOST) {

        } else {
            // We only allow allocating on device or unified if CUDA is enabled and devices are available
            if constexpr(jump::cuda_enabled()) {
                if(jump::devices_available()) {
                    if(location == memory_t::DEVICE) {
                        
                    } else {

                    }
                } else {
                    throw std::runtime_error("No devices available, unable to allocate on device or in unified memory");
                }
            } else {
                throw std::runtime_error("CUDA is not enabled, unable to allocate on device or in unified memory");
            }
        }
    }

    void deallocate() {
        if(size_ == 0) {
            return;
        }

        if(host_data_ != nullptr) {
            free(host_data_);
        }

        primary_location_ = memory_t::HOST;
        host_data_ = nullptr;
        device_data_ = nullptr;
        size_ = 0;
    }

private:
    //! The primary location for this bfufer
    memory_t primary_location_ = memory_t::HOST;
    //! Pointer to a memory buffer (if allocated to host (or unified)
    void* host_data_ = nullptr;
    //! Pointer to a memory buffer (if allocated to device (or unified)
    void* device_data_ = nullptr;
    //! Size of the array (x dim)
    std::size_t size_ = 0;

}; /* class memory_buffer */


} /* namespace jump */

#endif /* JUMP_MEMORY_BUFFER_HPP_ */
