#ifndef JUMP_SHARED_PTR_HPP_
#define JUMP_SHARED_PTR_HPP_

// JUMP
#include <jump/device_interface.hpp>
#include <jump/memory_buffer.hpp>

namespace jump {

/**
 * @brief a single-dimension array of a particular type
 * @tparam T the type of the array
 */
template<typename T>
class shared_ptr {
public:
    /**
     * @brief we do allow a default constructor for the array
     *  type, however; it cannot allocate data except through assignment
     */
    shared_ptr() {
    }

    /**
     * @brief rather than using a raw pointer this shared_ptr
     *  type uses the memory buffer as the tracking / storage
     *  mechanism since it can encapsalate all complex memory
     *  management logic in one place.
     * @param buf the memory buffer containing the object
     */
    shared_ptr(const memory_buffer& buf):
        buffer_(buf)
    {
    }

    /**
     * @brief copy-construct a shared_ptr
     * @param ptr the shared_ptr to copy from
     * @note following this operation the new shared_ptr and ptr
     *   will share ownership of the same shared object
     */
    shared_ptr(const shared_ptr& ptr):
        buffer_(ptr.buffer_)
    {}

    /**
     * @brief move-construct a shared_ptr
     * @param ptr the shared_ptr to transfer ownership from
     */
    shared_ptr(shared_ptr&& ptr):
        buffer_(std::move(ptr.buffer_))
    {}

    /**
     * @brief copy-assign from another shared_ptr
     * @param ptr the shared_ptr to copy from
     * @return shared_ptr& this shared_ptr that now shares
     *  ownership of data with ptr
     */
    shared_ptr& operator=(const shared_ptr& ptr) {
        dereference_buffer();
        buffer_ = ptr.buffer_;
        return *this;
    }

    /**
     * @brief move-assign from another shared_ptr
     * @param ptr the shared_ptr to move from
     * @return shared_ptr& this shared_ptr that now has ownership
     *  of the data from ptr
     */
    shared_ptr& operator=(shared_ptr&& ptr) {
        dereference_buffer();
        buffer_ = std::move(ptr.buffer_);
        return *this;
    }

    /**
     * @brief dereference the buffer and deconstruct
     */
    ~shared_ptr() {
        dereference_buffer();
    }

    /**
     * @brief access the underlying pointer
     * @return a pointer to the shared object
     */
    JUMP_INTEROPABLE
    T* get() const {
        return buffer_.data<T>();
    }

    /**
     * @brief derereference the pointer
     */
    JUMP_INTEROPABLE
    T& operator*() const {
        return *get();
    }

    /**
     * @brief derereference the pointer
     */
    JUMP_INTEROPABLE
    T* operator->() const {
        return get();
    }

    /**
     * @brief transfer all data to device
     */
    void to_device() {
        if constexpr(class_interface<T>::to_device_defined()) {
            get()->to_device();
        }
        buffer_.to_device();
    }

    /**
     * @brief transfer all data from device
     */
    void from_device() {
        buffer_.from_device();
        if constexpr(class_interface<T>::from_device_defined()) {
            get()->from_device();
        }
    }

    /**
     * @brief sync any pointers for usage on device
     *  see memory_buffer::sync();
     */
    void sync() {
        buffer_.sync();
    }

    /**
     * @brief provide direct access to the underlying memory buffer
     * @return memory_buffer& the memory buffer containing array data
     */
    memory_buffer& buffer() {
        return buffer_;
    }
private:
    /**
     * @brief manually perform buffer dereferencing
     *  to ensure descruction happens properly
     */
    void dereference_buffer() {
        buffer_.release([&](){
            get()->~T();
        });
    }

    //! The memory buffer containing the object
    memory_buffer buffer_;

}; /* class shared_ptr */

/**
 * @brief creates a shared pointer
 * @tparam T the type to create a shared_ptr of
 * @tparam Args the types for the arguments to the constructor
 * @param location the location to generate the shared object on
 *  (can be HOST or UNIFIED, does not support construction on device)
 * @param args the arguments to the object constructor
 * @return shared_ptr<T> a shared_ptr containing a new object of type T
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared_on(const memory_t& location, Args&&... args) {
    // NOTE: be careful to keep this construction logic in sync
    // with shared_ptr<T>::control_block::dereference() destruction logic
    if(location == memory_t::HOST || location == memory_t::UNIFIED) {
        // First we generate the memory buffer for it to exist on
        auto buf = memory_buffer();
        buf.allocate<T>(1, location);
        // Then we construct the object in that memory
        new(buf.data<T>()) T(args...);
        // lastly we construct the shared pointer around that buffer
        return shared_ptr<T>(buf);
    }
    throw std::runtime_error("Unable to construct a shared pointer directly on device");
}

/**
 * @brief creates a shared_ptr on HOST memory, equivalent
 *  to make_shared_on(memory_t::HOST, args...);
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared(Args&&... args) {
    return make_shared_on<T>(memory_t::HOST, args...);
}

/**
 * @brief creates a shared_ptr on HOST memory, equivalent
 *  to make_shared_on(memory_t::HOST, args...);
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared_on_host(Args&&... args) {
    return make_shared_on<T>(memory_t::HOST, args...);
}

/**
 * @brief creates a shared_ptr on UNIFIED memory, equivalent
 *  to make_shared_on(memory_t::UNIFIED, args...);
 */
template<typename T, typename... Args>
inline shared_ptr<T> make_shared_on_unified(Args&&... args) {
    return make_shared_on<T>(memory_t::UNIFIED, args...);
}

} /* namespace jump */

#endif /* JUMP_SHARED_PTR_HPP_ */
