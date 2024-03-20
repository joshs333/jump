/**
 * @file array.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief adds an array type that can be used interopably
 * @date 2023-03-17
 */
#ifndef JUMP_ARRAY_HPP_
#define JUMP_ARRAY_HPP_

// JUMP
#include <jump/device_interface.hpp>
#include <jump/memory_buffer.hpp>

namespace jump {


/**
 * @brief a simple helper class to handle comma
 *  assignment of values into arrays
 * @tparam T the value type to be assigned
 */
template<typename T, std::size_t _max_dims = 4>
class array_comma_helper {
public:
    //! Convenience type for the value type contained
    using value_type = T;

    /**
     * @brief construct the helper
     * @param start_data beginning of data to fill (not assignable)
     * @param end_data end of data to fill
     */
    JUMP_INTEROPABLE
    array_comma_helper(T* start_data, T* end_data):
        start_data_(start_data),
        end_data_(end_data)
    {}

    /**
     * @brief assign into array using comma operator
     * @param value the value to assign
     */
    JUMP_INTEROPABLE
    array_comma_helper& operator,(const T& value) {
        #if JUMP_ON_DEVICE
            #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                assert(start_data_ < end_data_ && "Unable to assign into array beyond it's range");
            #endif
        #else
            if(start_data_ >= end_data_)
                throw std::out_of_range("Unable to assign into array beyond it's range");
        #endif
        *start_data_ = value;
        ++start_data_;
        return *this;
    }

private:
    //! Start data (where to assign)
    T* start_data_;
    //! End data (cannot assign here or beyond)
    T* end_data_;

}; /* class array_comma_helper */


/**
 * @brief a single-dimension array of a particular type
 * @tparam T the type of the array
 */
template<typename T>
class array {
public:
    //! Convenience type for the value type contained
    using value_type = T;
    //! Type used to express size
    using size_type = std::size_t;
    //! Type used to express size difference
    using difference_type = std::ptrdiff_t;
    //! Convenience typedef for a reference to the underlying type
    using reference = T&;
    //! Convenience typedef for a const-reference to the underlying type
    using const_reference = const T&;
    //! Convenience typedef for a pointer to the underlying type
    using pointer = T*;
    //! Convenience typedef for a const pointer to the underlying type
    using const_pointer = const T*;

    /**
     * @brief an iterator over a continuous buffer
     *  that is iteropable but can NOT cross the
     *  CPU / GPU barrier safely.
     */
    struct iterator {
        //! Currently implements forward iterator
        using iterator_category = std::forward_iterator_tag;
        //! Same as array::difference_type
        using difference_type   = array::difference_type;
        //! Same as array::value_type
        using value_type        = array::value_type;
        //! Same as array::pointer
        using pointer           = array::pointer;
        //! Same as array::reference
        using reference         = array::reference;

        /**
         * @brief construct a new iterator
         * @param data the data to point to
         */
        iterator(
            pointer data
        ):
            data_(data)
        {}

        /**
         * @brief convert iterator to a reference
         * @return reference to the current object
         */
        reference operator*() const {
            return *data_;
        }

        /**
         * @brief convert iterator to a pointer
         * @return pointer to the current object
         */
        pointer operator->() {
            return data_;
        }

        /**
         * @brief increment the iterator
         * @return this iterator by reference (now incremented)
         */
        iterator& operator++() {
            data_++;
            return *this;
        }

        /**
         * @brief increment the iterator
         * @return an interator still pointing to the previous object
         */
        iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        /**
         * @brief eq operator between iterators
         * @param a one iterator to compare
         * @param b another iterator to compare
         * @return true if same, false if not
         */
        friend bool operator== (const iterator& a, const iterator& b) {
            return a.data_ == b.data_;
        };

        /**
         * @brief neq operator between iterators
         * @param a one iterator to compare
         * @param b another iterator to compare
         * @return true if a != b, false if not
         */
        friend bool operator!= (const iterator& a, const iterator& b) {
            return a.data_ != b.data_;
        };

        //! The current position pointer in the buffer
        pointer data_;

    }; /* struct iterator */

    /**
     * @brief A small header to an array buffer containing additional information
     *  probably doesn't need to be make into a struct since it only contains one
     *  value, but i want to make it easier to add more later just in case.
     */
    struct buffer_header {
        //! The size of the array (in sizes of T)
        std::size_t size;
        //! The base number of reservations
        std::size_t reservation_size;
    };

    //! Tag to force the array to construct without explicitly initializing values
    struct no_init {};

    /**
     * @brief construct a new array object, don't
     *  initialize any objects (start as empty array)
     * @param location the memory type to allocate on
     */
    array(const memory_t& location = memory_t::HOST) {
        allocate(0, location);
    }

    /**
     * @brief construct a new array object, default
     *  construct values in the array
     * @param size the size of the array to construct
     * @param location the memory type to allocate on
     */
    array(const std::size_t& size, const memory_t& location = memory_t::HOST) {
        allocate(size, location);

        for(int i = 0; i < size; ++i) {
            new(&buffer_.data<T>(sizeof(buffer_header))[i]) T();
        }
    }

    /**
     * @brief Cconstruct a new array object without value initialization
     * @param size the size of the array
     * @param location the memory type to allocate on
     */
    array(const std::size_t& size, const no_init&, const memory_t& location = memory_t::HOST) {
        allocate(size, location);
    }

    /**
     * @brief construct a new array object, construct values
     *  from a default value
     * @param size the size of the array to construct
     * @param default_value the default value to set objects in the array to
     * @param location the memory type to allocate on
     */
    array(const std::size_t& size, const T& default_value, const memory_t& location = memory_t::HOST) {
        allocate(size, location);

        for(int i = 0; i < size; ++i) {
            new(&buffer_.data<T>(sizeof(buffer_header))[i]) T(default_value);
        }
    }


    /**
     * @brief construct a new array object from values
     * @param values the values to initialize from
     * @param location the memory type to allocate on
     */
    array(const std::initializer_list<T>& values, const memory_t& location = memory_t::HOST) {
        allocate(values.size(), location);

        auto idx = 0;
        for(auto& v : values) {
            new(&buffer_.data<T>(sizeof(buffer_header))[idx++]) T(v);
        }
    }

    /**
     * @brief copy-construct an array
     * @param ar the array to copy from
     * @note following this operation the new array and ar
     *   will share ownership of the same array data.
     */
    array(const array& ar):
        buffer_(ar.buffer_)
    {}

    /**
     * @brief move-construct an array
     * @param ar the array to move from
     */
    array(array&& ar):
        buffer_(std::move(ar.buffer_))
    {}

    /**
     * @brief copy-assign from another array
     * @param ar the array to copy from
     * @return array& this array that now shares
     *  ownership of data with ar
     */
    array& operator=(const array& ar) {
        dereference_buffer();
        buffer_ = ar.buffer_;
        return *this;
    }

    /**
     * @brief move-assign from another array
     * @param ar the array to move from
     * @return array& this array that now has ownership
     *  of the data from ar
     */
    array& operator=(array&& ar) {
        dereference_buffer();
        buffer_ = std::move(ar.buffer_);
        return *this;
    }

    /**
     * @brief dereference the buffer and deconstruct
     */
    ~array() {
        dereference_buffer();
    }

    /**
     * @brief set the reservation size (the fundamental memory allocation / reservation block)
     * @param res the new reservation size
     */
    void set_reservation_size(const std::size_t& res) {
        buffer_.data<buffer_header>()->reservation_size = res;
    }

    /**
     * @brief gets the reservation size that is currently set
     * @return std::size_t the reservation size
     */
    std::size_t reservation_size() {
        return buffer_.data<buffer_header>()->reservation_size;
    }

    /**
     * @brief reserve space in the array
     * @param new_size the new size to reserve
     */
    void reserve(const std::size_t& new_size) {
        if(new_size == size()) return;
        if(new_size < size()) {
            for(std::size_t i = new_size - 1; i < size(); ++i) {
                at(i).~T();
            }
            buffer_.data<buffer_header>()->size = new_size;
        }
        buffer_.reallocate((new_size * sizeof(T)) + sizeof(buffer_header));
    }

    /**
     * @brief remove all elements from the container (does not re-allocate)
     */
    void clear() {
        if(!buffer_.data<buffer_header>()) {
            return;
        }
        for(std::size_t i = 0; i < size(); ++i) {
            at(i).~T();
        }
        buffer_.data<buffer_header>()->size = 0;
    }

    /**
     * @brief gets the size of buffer (the total number of
     *  elements that can be stored without allocation)
     */
    JUMP_INTEROPABLE
    std::size_t capacity() const {
        return (buffer_.size() - sizeof(buffer_header)) / sizeof(T);
    }

    /**
     * @brief get the size of the array
     */
    JUMP_INTEROPABLE
    std::size_t size() const {
        if(!buffer_.data<buffer_header>()) {
            return 0;
        }
        return buffer_.data<buffer_header>()->size;
    }

    /**
     * @brief insert an object at the end of the array
     * @param object the object to add
     */
    void push_back(const T& object) {
        if(size() == capacity()) {
            reserve((size() / buffer_.data<buffer_header>()->reservation_size + 1) * buffer_.data<buffer_header>()->reservation_size);
        }

        new (&buffer_.data<T>(sizeof(buffer_header))[buffer_.data<buffer_header>()->size++]) T(object);
    }

    /**
     * @brief access array members
     * @param index the index to access
     * @return the array member by reference
     */
    JUMP_INTEROPABLE
    inline T& at(const std::size_t& index) {
        #if JUMP_ON_DEVICE
            #if JUMP_ASSERT_SIZE_ON_DEVICE
                assert(index < size() && "jump::array index must be less than size");
            #endif
        #else
            if(index >= size()) {
                throw std::out_of_range("jump::array index " + std::to_string(index) + " >= " + std::to_string(size()) );
            }
        #endif
        return buffer_.data<T>(sizeof(buffer_header))[index];
    }

    /**
     * @brief access array members
     * @param index the index to access
     * @return the array member by const reference
     */
    JUMP_INTEROPABLE
    inline T& at(const std::size_t& index) const {
        #if JUMP_ON_DEVICE
            #if JUMP_ASSERT_SIZE_ON_DEVICE
                assert(index < size() && "jump::array index must be less than size");
            #endif
        #else
            if(index >= size()) {
                throw std::out_of_range("jump::array index " + std::to_string(index) + " >= " + std::to_string(size()) );
            }
        #endif
        return buffer_.data<T>(sizeof(buffer_header))[index];
    }

    /**
     * @brief index operator
     * @param index the index to access
     * @return T& value by reference at(index);
     */
    JUMP_INTEROPABLE
    T& operator[](const std::size_t& index) {
        return at(index);
    }

    /**
     * @brief index operator
     * @param index the index to access
     * @return T& value by const reference at(index);
     */
    JUMP_INTEROPABLE
    T& operator[](const std::size_t& index) const {
        return at(index);
    }

    /**
     * @brief assign into multi_array_view using istream and comma operator
     * @param value the first value to assign
     * @return a comma helper that can take values via the comma operator to assign into the multi_array
     */
    JUMP_INTEROPABLE
    array_comma_helper<T> operator<<(const T& value) const {
        auto ptr = buffer_.data<T>(sizeof(buffer_header));
        return array_comma_helper<T>(ptr, ptr + size()), value;
    }

    /**
     * @brief get a begin iterator
     * @return iterator that points to the first element
     */
    iterator begin() const {
        return iterator(buffer_.data<T>(sizeof(buffer_header)));
    }

    /**
     * @brief get an end iterator
     * @return iterator that points just past the last element :)
     */
    iterator end() const {
        return iterator(buffer_.data<T>(sizeof(buffer_header)) + size());
    }

    /**
     * @brief transfer all data to device
     */
    void to_device() {
        if constexpr(class_interface<T>::to_device_defined()) {
            for(std::size_t i = 0; i < size(); ++i) {
                at(i).to_device();
            }
        }
        buffer_.to_device();
    }

    /**
     * @brief transfer all data from device
     */
    void from_device() {
        buffer_.from_device();
        if constexpr(class_interface<T>::from_device_defined()) {
            for(std::size_t i = 0; i < size(); ++i) {
                at(i).from_device();
            }
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
    JUMP_INTEROPABLE
    memory_buffer& buffer() {
        return buffer_;
    }

    /**
     * @brief provide direct access to the underlying memory buffer
     * @return const memory_buffer& the memory buffer containing array data
     */
    JUMP_INTEROPABLE
    const memory_buffer& buffer() const {
        return buffer_;
    }

    /**
     * @brief access the primary location of this array
     * @return the main location memory is allocated on (if HOST, may also have a DEVICE segment)
     */
    memory_t location() const {
        return buffer_.location();
    }

private:
    /**
     * @brief internal method to isolate where the allocation is done
     *  across all constructors, after this they can initialize values
     *  as the constructor dictates
     */
    void allocate(const std::size_t& size, const memory_t& location) {
        if(location == memory_t::DEVICE)
            throw std::runtime_error("jump::array does not currently support construction with location == memory_t::DEVICE");

        auto alloc_size = size;
        if(size == 0)
            alloc_size = 100;

        buffer_.allocate((alloc_size * sizeof(T)) + sizeof(buffer_header), location);

        buffer_.data<buffer_header>()->size = size;
        buffer_.data<buffer_header>()->reservation_size = 100;
    }

    /**
     * @brief manually perform buffer dereferencing
     *  to ensure descruction happens properly
     */
    void dereference_buffer() {
        buffer_.release([&]() {
            for(std::size_t i = 0; i < size(); ++i) {
                at(i).~T();
            }
        });
    }

    //! The memory buffer containing all the data
    memory_buffer buffer_;

}; /* class array */

} /* namespace jump */

#endif /* JUMP_ARRAY_HPP_ */
