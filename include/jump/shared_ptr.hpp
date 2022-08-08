/**
 * @file shared_ptr.hpp
 * @author Joshua Spisak
 * @brief implementation of jump::shared_ptr an interopable shared_ptr data structure
 * @date 2022-05-07
 */
#ifndef JUMP_SHARED_PTR_HPP_
#define JUMP_SHARED_PTR_HPP_

#include <jump/device_interface.hpp>

#include <stdexcept>

//! top-level utils namespace
namespace jump {

//! An internal namespace for the shared_ptr class to use
namespace shared_ptr_helpers {

#ifdef JUMP_ENABLE_CUDA
    /**
     * @brief constructs an object that lives in device memory
     * @tparam T type of object
     * @tparam Args variadic arguments for constructor
     * @param pointer pre-allocated section of device memory to construct T on
     * @param args arguments to constructor
     * @note IMPORTANT: args must all be mem-copyable without any issues (no host pointers, etc...)
     * @note IMPORTANT: must be called with <<<1,1>>>
     */
    template<typename T, typename... Args>
    __global__ void deviceConstructObject(T* pointer, Args... args) {
        new(pointer) T(args...);
    }

    /**
     * @brief destruct an object that lives in device memory
     * @tparam T type of object
     * @param pointer where in device memory this object lives
     * @note IMPORTANT: must be called with <<<1,1>>>
     * @note this does not free the memory... just called the destructor
     */
    template<typename T>
    __global__ void deviceDestructObject(T* pointer) {
        pointer->~T();
    }
#endif

} /* namespace shared_ptr_helpers */

/**
 * @brief manages a pointer to make sure memory is handled correctly
 *  while allowing shared access
 * @tparam T type of the managed pointer
 */
template<typename T>
class shared_ptr {
    //! Internal control_block
    struct control_block {
        using count_t = unsigned int;

        //! Pointer to the object
        T* pointer;
        //! Count references to this shared object
        count_t use_count;
        //! Counts objects watching this control block
        count_t observer_count;
        //! Pointer to data on device (when applicable)
        T* device_pointer;
        //! Where the pointer lives primarily (effects pointer behavior)
        jump::memory location;

        /**
         * @brief creates a new control block, marks an observer and a user (if pointer is valid)
         * @param unowned_pointer new pointer that is not owned by any other objects
         **/
        control_block(jump::memory pointer_location, T* unowned_pointer):
            pointer(nullptr),
            use_count(1),
            observer_count(1),
            device_pointer(nullptr),
            location(pointer_location)
        {
            if(location == jump::memory::DEVICE) {
                device_pointer = unowned_pointer;
            } else {
                if(location == jump::memory::UNIFIED) {
                    device_pointer = unowned_pointer;
                }
                pointer = unowned_pointer;
            }
        }

        count_t dereference() {
            if(--use_count <= 0) {
                if(location == jump::memory::UNIFIED) {
                    #ifdef JUMP_ENABLE_CUDA
                        device_pointer->~T();
                        cudaFree(device_pointer);
                        device_pointer = nullptr;
                        pointer = nullptr;
                    #else
                        std::fprintf(stderr, "ERROR: dereferencing unified shared_ptr, but CUDA is not available, ensure full project is compiled with CUDA.\n");
                    #endif
                } else {
                    if(pointer != nullptr) {
                        delete pointer;
                        pointer = nullptr;
                    }
                    if(device_pointer != nullptr) {
                        #ifdef JUMP_ENABLE_CUDA
                            // Seems like usually the destructor is not __device__, not sure if
                            // the most sensible default is to not destruct on DEVICE or to be
                            // super clear about reqs. on DEVICE pointers (which probably won't
                            // be used that much, HOST -> toGPU() will probably be more common)
                            // if(location == jump::memory::DEVICE) {
                            //     shared_ptr_helpers::deviceDestructObject(device_pointer);
                            // }
                            cudaFree(device_pointer);
                            device_pointer = nullptr;
                        #else
                            std::fprintf(stderr, "ERROR: device_pointer != nullptr, but CUDA is not available, ensure full project is compiled with CUDA.\n");
                        #endif
                    }
                }
            }
            return --observer_count;
        }

        count_t unobserve() {
            return --observer_count;
        }

        void reference() {
            ++observer_count;
            ++use_count;
        }

        void observe() {
            ++observer_count;
        }
    };

public:
    using count_t = typename control_block::count_t;

    /**
     * @brief Construct a new shared ptr object (on HOST, nullptr)
     */
	shared_ptr():
        block_(new control_block(jump::memory::HOST, nullptr)),
        device_pointer_(block_->device_pointer)
    {}

    /**
     * @brief Construct a new shared ptr object
     * @param pointer the unowned pointer to take ownership of
     */
	shared_ptr(T* pointer):
        block_(new control_block(jump::memory::HOST, pointer)),
        device_pointer_(block_->device_pointer)
    {}

    /**
     * @brief Construct a new shared ptr object
     * @param location what memory the pointer lives in
     * @param pointer the unowned pointer to take ownership of
     */
	shared_ptr(jump::memory location, T* pointer):
        block_(new control_block(location, pointer)),
        device_pointer_(block_->device_pointer)
    {}

    /**
     * @brief construct from another shared pointer
     * @param obj the other shared pointer containing the object we should also
     */
	shared_ptr(const shared_ptr& obj) {
        block_ = obj.block_;
        block_->reference();
        device_pointer_ = block_->device_pointer;
	}

    /**
     * @brief move-construct from another shared pointer
     * @param obj the other shared pointer we want to ownership
     */
    shared_ptr(shared_ptr&& obj) {
        block_ = obj.block_;
        obj.block_ = new control_block(jump::memory::HOST, nullptr);
        device_pointer_ = block_->device_pointer;
	}

    /**
     * @brief assign a different target object
     * @param obj a shared_ptr we want to copy from
     * @return this shared pointer
     */
	shared_ptr& operator=(const shared_ptr& obj) {
        dereference_block();
        block_ = obj.block_;
        block_->reference();
        device_pointer_ = block_->device_pointer;
        return *this;
	}

    /**
     * @brief move-assignment a different target object
     * @param obj a shared_ptr we want to take ownership from
     * @return this shared pointer
     */
	shared_ptr& operator=(shared_ptr&& obj) {
        dereference_block();
        block_ = obj.block_;
        obj.block_ = new control_block(jump::memory::HOST, nullptr);
        device_pointer_ = block_->device_pointer;
        return *this;
	}

    /**
     * @brief gets the number of pointers sharing ownership of this object
     * @return unsigned int the number of pointers sharing ownership
     */
	unsigned int use_count() const {
        return block_->use_count;
	}

    /**
     * @brief gets the number of pointers observing of this object (weak + shared pointers)
     * @return unsigned int the number of pointers observing this object
     */
	unsigned int observer_count() const {
        return block_->observer_count;
	}

    /**
     * @brief getter for object (safe for Host / Device)
     * @return pointer to object
     */
    JUMP_INTEROPABLE
	T* get() const {
        #if ON_GPU
            return device_pointer_;
        #else
            return block_->pointer;
        #endif
	}

    /**
     * @brief getter for object on device (safe for Host / Device)
     * @return pointer to object in device or unified memory
     */
    JUMP_INTEROPABLE
	T* get_device() const {
        return device_pointer_;
	}

    /**
     * @brief getter for object on device (safe for Host / Device)
     * @return pointer to object in device or unified memory
     */
    JUMP_INTEROPABLE
	T* operator->() const {
        #if ON_GPU
            return device_pointer_;
        #else
            return block_->pointer;
        #endif
	}

    /**
     * @brief operator to dereference pointer (safe for Host / Device)
     * @return object passed by reference in device or unified memory
     */
    JUMP_INTEROPABLE
	T& operator*() const {
        #if ON_GPU
            return *(device_pointer_);
        #else
            return *(block_->pointer);
        #endif
	}

    /**
     * @brief check pointer validity (safe for Host / Device)
     * @return true if pointer is valid, false if not
     */
    JUMP_INTEROPABLE
    explicit operator bool() const {
        #if ON_GPU
            return (device_pointer_ != nullptr);
        #else
            return (block_->pointer != nullptr);
        #endif
    }

    /**
     * @brief destroy the shared ptr object (destroy any shared state if last owner)
     */
	~shared_ptr() {
        #if not ON_GPU
            dereference_block();
        #endif
	}

    /**
     * @brief transfer data from HOST to DEVICE
     *  (no effect if UNIFIED, throws execption if initialized in DEVICE)
     */
    void toGPU() {
        #ifdef JUMP_ENABLE_CUDA
            if (block_->location == jump::memory::DEVICE) {
                if(block_->device_pointer == nullptr) {
                    return;
                }
                throw std::runtime_error("Unable to transfer to GPU when location is DEVICE");
            } else if(block_->location == jump::memory::HOST) {
                if(block_->pointer == nullptr) {
                    return;
                }
                if(block_->device_pointer == nullptr) {
                    cudaMalloc(&block_->device_pointer, sizeof(T));
                }

                bool copy = true;
                if constexpr(jump::device_interface<T>::to_)
                    block_->pointer->toGPU();
                if(copy)
                    cudaMemcpy(block_->device_pointer, block_->pointer, sizeof(T), cudaMemcpyHostToDevice);

                device_pointer_ = block_->device_pointer;
            } else if(block_->location == jump::memory::UNIFIED) {
                if constexpr(shared_ptr_helpers::toGPUDefined<T>{})
                    block_->pointer->toGPU(nullptr);
                else if constexpr(shared_ptr_helpers::toGPUVoidDefined<T>{})
                    block_->pointer->toGPU();
            }
        #else
            throw std::runtime_error("Unable to transfer to GPU when CUDA is not available.");
        #endif
    }

    /**
     * @brief transfer data from HOST to DEVICE
     *  (no effect if UNIFIED, throws execption if initialized in DEVICE)
     */
    void fromGPU() {
        device_pointer_ = block_->device_pointer;
        #ifdef JUMP_ENABLE_CUDA
            if (block_->location == jump::memory::DEVICE) {
                if(block_->device_pointer == nullptr) {
                    return;
                }
                throw std::runtime_error("Unable to transfer from GPU when location is DEVICE");
            } else if(block_->location == jump::memory::HOST) {
                if(block_->pointer == nullptr) {
                    return;
                }
                if(block_->device_pointer == nullptr) {
                    throw std::runtime_error("DEVICE memory is not allocated, unable to copy from GPU");
                }
                bool copy = true;
                if constexpr(shared_ptr_helpers::fromGPUDefined<T>{})
                    copy = block_->pointer->fromGPU(block_->device_pointer);
                else if constexpr(shared_ptr_helpers::fromGPUVoidDefined<T>{})
                    block_->pointer->fromGPU();
                if(copy)
                    cudaMemcpy(block_->pointer, block_->device_pointer, sizeof(T), cudaMemcpyDeviceToHost);

                device_pointer_ = block_->device_pointer;
            } else if(block_->location == jump::memory::UNIFIED) {
                if constexpr(shared_ptr_helpers::fromGPUDefined<T>{})
                    block_->pointer->fromGPU(nullptr);
                else if constexpr(shared_ptr_helpers::fromGPUVoidDefined<T>{})
                    block_->pointer->fromGPU();
            }
        #else
            throw std::runtime_error("Unable to transfer to GPU when CUDA is not available.");
        #endif
    }

    /**
     * @brief makes sure pointer is ready for direct copy to GPU
     *  recommended to call before any GPU operations
     *  (unless toGPU() is already callled)
     */
    void sync() {
        device_pointer_ = block_->device_pointer();
    }

private:
    //! Dereference block and free if last observer
    void dereference_block() {
        auto observers = block_->dereference();
        if(observers <= 0)
            delete block_;
        block_ = nullptr;
    }

    //! The control block storing shared state between pointers
    control_block* block_;

    /**
     * @brief Pointer to object in device memory, this allows the shared_ptr
     *  To be memcopied to GPU and just work, even source of truth on
     *  GPU pointer state is in the control_block (hence the need to sync()).
     * 
     * The motivation for the separate pointer is allowing late transfer to
     * device if the object is initially just a host pointer. If the pointer
     * is stored solely in the shared_ptr class it must exist and remain unchanged
     * in the initial shared_ptr to be copied consistently across any other
     * shared_ptrs for this object. We could also introduce another level of
     * indirection and allocate a section of UNIFIED or DEVICE memory to store
     * the real pointer to the object that could be created later... but that
     * felt more complex and could harm GPU performance with pointers across
     * different sections of memory from this initial reference.
     * 
     * It would also potentially complicate the memory scheme when using
     * UNIFIED memory instead of split HOST / DEVICE memory.
     **/
    T* device_pointer_;
}; /* class shared_ptr */

/**
 * @brief contructs a shared pointer
 * @tparam T type to construct a shared pointer of
 * @tparam Args variadic args for T constructur
 * @param location what segment of memory to create the object (host, unified)
 * @param args arguments to construct the new T
 * @return shared_ptr<T> to the new T
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared_on(jump::memory location, Args&&... args) {
    // NOTE: be careful to keep this construction logic in sync
    // with shared_ptr<T>::control_block::dereference() destruction logic
    if(location == jump::memory::HOST) {
        return shared_ptr<T>(jump::memory::HOST, new T(args...));
    } else if(location == jump::memory::UNIFIED) {
        #ifdef JUMP_ENABLE_CUDA
            T* pointer;
            cudaMallocManaged(&pointer, sizeof(T));
            return shared_ptr<T>(jump::memory::UNIFIED, new(pointer) T(args...));
        #else
            throw std::runtime_error("CUDA is not available, unable to create shared_ptr except on host.");
        #endif
    } else {
        throw std::runtime_error("unabled to create a shared_ptr on DEVICE directly.");
    }
}

/**
 * @brief contructs a shared pointer on the host (normal shared_ptr) see make_shared_on()
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared(Args&&... args) {
    return make_shared_on<T>(jump::memory::HOST, args...);
}

/**
 * @brief contructs a shared pointer on the host (same as make_shared())
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared_on_host(Args&&... args) {
    return make_shared_on<T>(jump::memory::HOST, args...);
}

/**
 * @brief contructs a shared pointer in unified memory
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared_on_unified(Args&&... args) {
    return make_shared_on<T>(jump::memory::UNIFIED, args...);
}

} /* namespace jump */

#endif /* JUMP_UTIL_SHARED_PTR_HPP_ */
