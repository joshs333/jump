/**
 * @file memory_buffer.hpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief object containing a raw buffer of memory and controlling it's
 *  location in either host or device memory.
 * @date 2022-10-05
 */
#ifndef JUMP_MEMORY_BUFFER_HPP_
#define JUMP_MEMORY_BUFFER_HPP_

// STD
#include <cstddef>
#include <stdexcept>
#include <cstdio>
#include <type_traits>
#include <functional>
#include <cstring>
#include <string_view>

// APPLICATION
#include <jump/device_interface.hpp>

//! Just Multi-Processing namespace
namespace jump {

//! Types of memory that can be allocated / managed
enum class memory_t : uint8_t {
    //! Memory allocated to the host in the heap
    HOST = 0,
    //! Memory allocated to a particular GPU device
    DEVICE = 1,
    //! Memory allocated to unified memory, available to HOST and DEVICE
    UNIFIED = 2,
    //! Max value for memory_t (UNKNOWN memory_t)
    UNKNOWN = 3
};

//! String representations for memory_t
static std::string_view memory_t_str[] = {
    "HOST",
    "DEVICE",
    "UNIFIED",
    "UNKNOWN"
};

/**
 * @brief an object to own and manage a raw memory buffer,
 *  by default copied instances of a memory buffer point
 *  to the same underlying buffer, only by constructing
 *  new memory_buffer isntances or calling memory_buffer.copy()
 *  can new buffers be generated.
 *
 *  This will serve as the base for the shared_ptr (a shared buffer
 *  containing a single object) and array types (a shared buffer
 *  containing multiple objects of a single class type).
 */
class memory_buffer {
public:
    //! Type for a custom reallocation function that can be called on realloc()
    using custom_realloc_func = std::function<void(void*,void*,std::size_t)>;

    //! The type used to count references
    using count_t = unsigned int;
    /**
     * @brief the control block stores all state that is shared
     *  between all instances of a memory_buffer pointing to the
     *  same underlying buffer.
     */
    struct control_block {
        //! A pointer to the buffer on the host (or unified)
        void* host_data;
        //! A pointer to the buffer on the device
        void* device_data;
        //! The size of the buffer
        std::size_t size;
        //! The number of references to this buffer
        count_t reference_counter;
        //! The primary location of the buffer (effects copy operations)
        memory_t location;

        /**
         * @brief Construct a new control block object, sets
         *  defaults and initializes reference counting
         */
        control_block():
            host_data(nullptr),
            device_data(nullptr),
            size(0),
            reference_counter(1),
            location(memory_t::HOST)
        {}

        /**
         * @brief performs an allocation of memory
         * @note if this control block already points to buffers it will call
         *  free_data() before performing any allocation functions
         * @note if buffer_size == 0, we will refuse to allocate
         * @param buffer_size the size of the buffer to allocate
         * @param location the location to allocate the buffer
         */
        void allocate(std::size_t buffer_size, memory_t location) {
            if(host_data != nullptr || device_data != nullptr) {
                deallocate();
            }
            if(buffer_size == 0) {
                return;
            }
            this->location = location;

            if(location == memory_t::HOST) {
                host_data = malloc(buffer_size);
            } else {
                // We only allow allocating on device or unified if CUDA is enabled and devices are available
                #ifdef JUMP_ENABLE_CUDA
                    if(jump::devices_available()) {
                        if(location == memory_t::DEVICE) {
                            auto error = cudaMalloc(&device_data, buffer_size);
                            if(error != cudaSuccess) {
                                throw cuda_error_exception(error);
                            }
                        } else if(location == memory_t::UNIFIED) {
                            auto error = cudaMallocManaged(&device_data, buffer_size);
                            host_data = device_data;
                            if(error != cudaSuccess) {
                                throw cuda_error_exception(error);
                            }
                        } else {
                            throw std::runtime_error("WTF?? UNKNOWN MEMORY_T");
                        }
                    } else {
                        throw jump::no_devices_exception("unable to allocate on device or unified memory");
                    }
                #else
                    throw jump::no_cuda_exception("unable to allocate on device or unified memory");
                #endif
            }
            size = buffer_size;
        }

        /**
        * @brief resize the underlying buffer (must be an increased size)
        * @param buffer_size the new size of the buffer
        */
        void reallocate(std::size_t buffer_size) {
            if(buffer_size < size) {
                throw std::runtime_error("buffer reallocation must be to a higher size");
            }
            if(buffer_size == size) {
                return;
            }
            auto old_host_data = host_data;
            auto old_device_data = device_data;
            auto old_buffer_size = size;
            host_data = nullptr;
            device_data = nullptr;
            allocate(buffer_size, location);

            if(location == memory_t::HOST || location == memory_t::UNIFIED) {
                std::memcpy(host_data, old_host_data, old_buffer_size);

                if(location == memory_t::UNIFIED) {
                    #if JUMP_ENABLE_CUDA
                        cudaFree(old_host_data);
                    #else
                        throw jump::no_cuda_exception("cannot free unified memory");
                    #endif
                } else {
                    if(old_host_data) {
                        free(old_host_data);
                    }
                    if(old_device_data) {
                        #if JUMP_ENABLE_CUDA
                            cudaFree(old_device_data);
                        #else
                            throw jump::no_cuda_exception("cannot free device memory");
                        #endif
                    }
                }
            } else {
                #if JUMP_ENABLE_CUDA
                    auto e = cudaMemcpy(device_data, old_device_data, old_buffer_size, cudaMemcpyDeviceToDevice);
                    if(e != cudaSuccess) {
                        throw cuda_error_exception(e);
                    }
                #else
                    throw jump::no_cuda_exception("Unable to memcpy device memory");
                #endif
            }
        }

        /**
         * @brief releases all data pointed to by this control_block,
         *  and nullifies all pointers.
         */
        void deallocate() {
            if(location == memory_t::UNIFIED) {
                if(host_data != nullptr) {
                    #ifdef JUMP_ENABLE_CUDA
                        cudaFree(host_data);
                        host_data = nullptr;
                        device_data = nullptr;
                    #else
                        throw jump::no_cuda_exception("cannot free unified memory");
                    #endif
                }
            } else {
                if(host_data != nullptr) {
                    free(host_data);
                    host_data = nullptr;
                }
                if(device_data != nullptr) {
                    #ifdef JUMP_ENABLE_CUDA
                        cudaFree(device_data);
                        device_data = nullptr;
                    #else
                        throw jump::no_cuda_exception("cannot free device memory");
                    #endif
                }
            }
        }

        /**
         * @brief allows a memory_buffer referencing this control_block
         *  to mark that it is no longer paying attention to it.
         * @note if there are no more references to the buffer, that
         *  buffer will automatically be released.
         * @arg dealloc_func function to call before deallocation
         * @return count_t the remaining number of references to the buffer
         * @note if this function returns 0 it is important that the
         *  memory_buffer dereferencing this control_block free it.
         */
        count_t dereference(std::function<void(void)> dealloc_func = nullptr) {
            auto count = --reference_counter;
            if(count == 0 && host_data != nullptr) {
                if(dealloc_func) {
                    try {
                        dealloc_func();
                    } catch(std::exception& e) {
                        deallocate();
                        throw e;
                    }
                }
                deallocate();
            }
            return count;
        }

        /**
         * @brief add a reference to this control block
         */
        void reference() {
            ++reference_counter;
        }

        /**
         * @brief transfer data from the host to the device
         */
        void to_device() {
            // There is no data to transfer, just return
            if(size == 0) {
                return;
            }
            // We can only transfer if cuda is enabled and there are devices to transfer to
            #ifdef JUMP_ENABLE_CUDA
                if(!jump::devices_available()) {
                    throw jump::no_devices_exception("unable to copy from host to device");
                }
                if(location == memory_t::UNIFIED) {
                    // Do nothing, memory is already available on the device
                } else if(location == memory_t::DEVICE) {
                    // we only allow a copy if we have host data to copy from
                    if(host_data == nullptr) {
                        throw std::runtime_error("Host memory is not allocated, unable to copy to device");
                    }

                    auto err = cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
                    if(err != cudaSuccess)
                        throw jump::cuda_error_exception(err, "unable to copy from host to device");
                } else {
                    // if 
                    if(device_data == nullptr) {
                        auto err = cudaMalloc(&device_data, size);
                        if(err != cudaSuccess)
                            throw jump::cuda_error_exception(err, "unable to copy from host to device");
                    }
                    auto err = cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
                    if(err != cudaSuccess)
                        throw jump::cuda_error_exception(err, "unable to copy from host to device");
                }
            #else
                throw jump::no_cuda_exception("cannot transfer buffer to device");
            #endif
        }

        /**
         * @brief transfer data from the host to the device
         */
        void from_device() {
            // There is no data to transfer, just return
            if(size == 0) {
                return;
            }
            // We can only transfer if cuda is enabled and there are devices to transfer to
            #ifdef JUMP_ENABLE_CUDA
                if(!jump::devices_available()) {
                    throw jump::no_devices_exception("unable to copy from device to host (a)");
                }
                if(location == memory_t::UNIFIED) {
                    // Do nothing, memory is already available on the device
                } else if(location == memory_t::DEVICE) {
                    // if there is no host buffer, we allocate it
                    if(host_data == nullptr) {
                        host_data = malloc(size);
                    }

                    auto err = cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
                    if(err != cudaSuccess)
                        throw jump::cuda_error_exception(err, "unable to copy from device to host (b)");
                } else {
                    // if we don't have device data, we can't copy from it
                    if(device_data == nullptr) {
                        throw std::runtime_error("Device memory is not allocated, unable to copy from device");
                    }
                    auto err = cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
                    if(err != cudaSuccess)
                        throw jump::cuda_error_exception(err, "unable to copy from device to host (c)");
                }
            #else
                throw jump::no_cuda_exception("cannot transfer buffer from device");
            #endif
        }

    }; /* struct control_block */

    /**
     * @brief construct an empty memory buffer
     * @note creates a clean new empty buffer :)
     */
    memory_buffer():
        block_(new control_block()),
        device_data_(nullptr),
        device_size_(0)
    {}

    /**
     * @brief copy-construct a new memory_buffer
     * @param buf the existing memory_buffer to construct from
     * @note both of these objects reference the same underlying
     *  buffer and now share ownership
     */
    memory_buffer(const memory_buffer& buf):
        block_(buf.block_),
        device_data_(buf.device_data_),
        device_size_(buf.device_size_)
    {
        block_->reference();
    }

    /**
     * @brief move-construct a new memory_buffer
     * @param buf the buffer to move from
     * @note transfers ownership of the control
     *  block, nullifies buf
     */
    memory_buffer(memory_buffer&& buf):
        block_(buf.block_),
        device_data_(buf.device_data_),
        device_size_(buf.device_size_)
    {
        buf.block_ = nullptr;
        buf.device_data_ = nullptr;
        buf.device_size_ = 0;
    }

    /**
     * @brief move-assignment from another buffer, transfers
     *  ownership and nullifies buf
     * @param buf the buffer to move from
     * @return multi_array& this object
     */
    memory_buffer& operator=(memory_buffer&& buf) {
        block_ = buf.block_;
        device_data_ = buf.device_data_;
        device_size_ = buf.device_size_;
        buf.block_ = nullptr;
        buf.device_data_ = nullptr;
        buf.device_size_ = 0;
        return *this;
    }

    /**
     * @brief copy-assignment from another buffer
     * @param buf the buffer to copy from
     * @return memory_buffer& this object
     * @note after this, both memory_buffers share ownership
     *  of the same underlying buffer
     */
    memory_buffer& operator=(const memory_buffer& buf) {
        // dereference the current control block
        dereference();
        block_ = buf.block_;
        device_data_ = buf.device_data_;
        device_size_ = buf.device_size_;
        block_->reference();
        return *this;
    }

    /**
     * @brief destroy this memory buffer object and dereference
     *  the control block / underlying buffer
     */
    ~memory_buffer() {
        dereference();
    }

    /**
     * @brief access the buffer
     * @tparam T optional type to cast the buffer pointer to (default void)
     * @param offset the bytes to offset relative to the base pointer
     * @return T* pointer to the buffer
     */
    template<typename T = void>
    JUMP_INTEROPABLE
    T* data(const std::size_t& offset = 0) const {
        #if JUMP_ON_DEVICE
            return reinterpret_cast<T*>(static_cast<char*>(device_data_) + offset);
        #else
            return reinterpret_cast<T*>(static_cast<char*>(block_->host_data) + offset);
        #endif
    }

    /**
     * @brief access the buffer (specifically for the host)
     * @tparam T optional type to cast the buffer pointer to (default void)
     * @return T* pointer to the host buffer
     */
    template<typename T = void>
    T* host_data() const {
        return reinterpret_cast<T*>(block_->host_data);
    }

    /**
     * @brief access the buffer (specifically for the device)
     * @tparam T optional type to cast the buffer pointer to (default void)
     * @return T* pointer to the host buffer
     */
    template<typename T = void>
    JUMP_INTEROPABLE
    T* device_data() const {
        #if JUMP_ON_DEVICE
            return reinterpret_cast<T*>(device_data_);
        #else
            return reinterpret_cast<T*>(block_->device_data);
        #endif
    }

    /**
     * @brief allocates a section of memory at some memory location
     * @tparam T the type to allocate (default is void)
     * @param buffer_size size of the buffer to allocate (automatic * sizeof(T) if T != void)
     * @param location the memory location to allocate in (HOST, DEVICE, or UNIFIED)
     */
    template<typename T = void>
    void allocate(std::size_t buffer_size, memory_t location = memory_t::HOST) {
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

        block_->allocate(allocation_size, location);
        sync();
    }

    /**
     * @brief resize the underlying buffer (must be an increased size)
     * @param buffer_size the new size of the buffer
     */
    template<typename T = void>
    void reallocate(std::size_t buffer_size) {
        std::size_t allocation_size;
        if constexpr(std::is_void<T>::value) {
            allocation_size = buffer_size;
        } else {
            allocation_size = buffer_size * sizeof(T);
        }
        block_->reallocate(allocation_size);
        sync();
    }

    /**
     * @brief deallocates the memory in the underlying buffer
     * @note nullifies the data for all memory_buffers pointing
     *  to the same underlying buffer
     */
    void deallocate() {
        block_->deallocate();
        sync();
    }

    /**
     * @brief gets the number of memory_buffers pointing to
     *  the same underlying buffer
     * @return count_t the number of owners of the underlying buffer
     */
    count_t owners() const {
        return block_->reference_counter;
    }

    /**
     * @brief gets the size of the buffer
     * @return std::size_t the size of the buffer
     */
    JUMP_INTEROPABLE
    std::size_t size() const {
        #if JUMP_ON_DEVICE
            return device_size_;
        #else
            return block_->size;
        #endif
    }

    /**
     * @brief transfer the buffer to the device
     */
    void to_device() {
        block_->to_device();
        sync();
    }

    /**
     * @brief transfer the buffer from the device
     * 
     */
    void from_device() {
        block_->from_device();
        sync();
    }

    /**
     * @brief syncs device data from the control
     *  block for usage on the GPU, called as part of the
     *  to / from device process, but can be done separately
     *  to prevent redundant data copies
     */
    void sync() {
        if(!block_) return;
        device_data_ = block_->device_data;
        device_size_ = block_->size;
    }

    /**
     * @brief provide direct access to the control block
     * @return control_block* pointer to the control block
     */
    control_block* block() const {
        return block_;
    }

    /**
     * @brief allows higher-level classes to insert custom behavior
     *  to the dereferencing process. Specifically this calls block_
     *  ->dereference() with dealloc_func and then dereferences the block.
     * @param dealloc_func the function to call before freeing the buffer
     */
    void release(
        const std::function<void(void)>& dealloc_func
    ) {
        if(!block_)
            return;
        
        auto references = block_->dereference(dealloc_func);
        if(references == 0) {
            delete block_;
        }
        block_ = nullptr;
        device_data_ = nullptr;
        device_size_ = 0;
    }

private:
    /**
     * @brief internal method to dereference the current
     *  control block and free the memory if no other
     *  memory_buffer's are referencing that block
     */
    void dereference() {
        if(!block_)
            return;
        auto references = block_->dereference();
        if(references == 0) {
            delete block_;
        }
        block_ = nullptr;
        device_data_ = nullptr;
        device_size_ = 0;
    }

    //! The control block containing all information for this memory buffer
    control_block* block_;
    //! The device pointer (if valid), which allows the memory buffer to
    //! be directly copied to the GPU and work without issue :)
    void* device_data_;
    //! Stores the size for GPU operations
    std::size_t device_size_;

}; /* class memory_buffer */


} /* namespace jump */

#endif /* JUMP_MEMORY_BUFFER_HPP_ */
