/**
 * @file string_view.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief a basic interopable string implementation
 *  (WARNING: NOT ALL OPERATIONS ARE IMPLEMENTED, JUST BARE MINIMUM
 *  FOR WHAT I NEED.)
 * @date 2023-04-26
 * @note at some point I should look more at the RAPIDS string API,
 *  but for simple cases rn I think this is easier.
 */
#ifndef JUMP_STRING_HPP_
#define JUMP_STRING_HPP_

#include <jump/string_utils.hpp>
#include <jump/device_interface.hpp>
#include <jump/memory_buffer.hpp>

namespace jump {

/**
 * @brief GPU compatible minimal string implementation, does NOT
 *  have all the same functions as std::string, just enough to
 *  meet current functional requirements.
 *
 * @tparam char_t The character type to use
 */
template<typename char_t = char>
class basic_string {
public:
    //! Defines the value_type that is used
    using value_type = char_t;

    /**
     * @brief default constructor - produces an empty string
     */
    basic_string() {
    }

    /**
     * @brief constructs a string from a const char*
     * @param str the string to construct from
     * @param location the location to construct on
     */
    basic_string(
        const char_t* str,
        const memory_t& location = memory_t::HOST
    ) {
        allocate(str, location);
        copy_string(str);
    }

    /**
     * @brief construct from a std::string for compatibility
     * @param str the string to construct from
     * @param location the location to construct on
     */
    basic_string(
        const std::string& str,
        const memory_t& location = memory_t::HOST
    ) {
        allocate(str.data(), str.size(), location);
        copy_string(str.data());
    }

    /**
     * @brief copy-constructor
     * @param s copy from s to a new string
     */
    basic_string(const basic_string& s) {
        if(s.size() > 0) {
            allocate(s.data(), s.location());
            copy_string(s.data());
        }
    }

    /**
     * @brief move constructor
     * @param s move this to a new string (s is now nullified)
     */
    basic_string(basic_string&& s):
        buffer_(std::move(s.buffer_))
    {}

    /**
     * @brief copy-assignment
     * @param s the string to assign from
     */
    basic_string& operator=(const basic_string& s) {
        buffer_.deallocate();
        if(s.size() > 0) {
            allocate(s.data(), s.location());
            copy_string(s.data());
        }
        return *this;
    }

    /**
     * @brief move-assignment
     * @param s the string to assign from (is nullified)
     */
    basic_string& operator=(basic_string&& s) {
        buffer_ = std::move(s.buffer_);
        return *this;
    }

    /**
     * @brief convert to a std string
     * @return std::string a std string containing the same string as this object
     */
    operator std::string() const {
        return std::string(c_str());
    }

    /**
     * @brief gets the length of this string (same as size())
     * @return the length
     */
    JUMP_INTEROPABLE
    inline std::size_t len() const {
        if(buffer_.size() > 1)
            return buffer_.size() - 1;
        return 0;
    }

    /**
     * @brief gets the length of the string (same as len())
     * @return the length
     */
    JUMP_INTEROPABLE
    inline std::size_t size() const {
        return len();
    }

    /**
     * @brief operator for comparison of strings
     * @param a string_view to compare
     * @param b string_view to compare
     * @return true if a and b are the same, false if not
     */
    JUMP_INTEROPABLE
    friend bool operator==(const basic_string& a, const basic_string& b) noexcept {
        return string_utils::compare(a, b) == 0;
    }

    /**
     * @brief operator for neq comparison of strings
     * @param a string_view to compare
     * @param b string_view to compare
     * @return true if a and b are different, false if not
     */
    JUMP_INTEROPABLE
    friend bool operator!=(const basic_string& a, const basic_string& b) noexcept {
        return string_utils::compare(a, b) != 0;
    }

    /**
     * @brief convert to a c-str
     * @return char_t* a null-terminated string array
     */
    JUMP_INTEROPABLE
    char_t* c_str() const {
        return data();
    }

    /**
     * @brief access the string data 
     * @return pointer to the string data on host or device
     */
    JUMP_INTEROPABLE
    char_t* data() const {
        return buffer_.data<char_t>();
    }

    /**
     * @brief access the string data (on host specifically)
     * @return pointer to the string data on host
     */
    char_t* host_data() const {
        return buffer_.host_data<char_t>();
    }

    /**
     * @brief access the string data (on device specifically)
     * @return pointer to the stirng data on device
     */
    JUMP_INTEROPABLE
    char_t* device_data() const {
        return buffer_.device_data<char_t>();
    }

    /**
     * @brief access characters in the string
     * @param index the index to access
     * @return the character by reference
     */
    JUMP_INTEROPABLE
    inline char_t& at(const std::size_t& index) {
        #if JUMP_ON_DEVICE
            #if JUMP_ASSERT_SIZE_ON_DEVICE
                assert(index < size() && "jump::string index must be less than size");
            #endif
        #else
            if(index >= size()) {
                throw std::out_of_range("jump::string index " + std::to_string(index) + " >= " + std::to_string(size()) );
            }
        #endif
        return buffer_.data<char_t>()[index];
    }

    /**
     * @brief access characters in the string
     * @param index the index to access
     * @return the character by const reference
     */
    JUMP_INTEROPABLE
    inline char_t& at(const std::size_t& index) const {
        #if JUMP_ON_DEVICE
            #if JUMP_ASSERT_SIZE_ON_DEVICE
                assert(index < size() && "jump::string index must be less than size");
            #endif
        #else
            if(index >= size()) {
                throw std::out_of_range("jump::string index " + std::to_string(index) + " >= " + std::to_string(size()) );
            }
        #endif
        return buffer_.data<char_t>()[index];
    }

    /**
     * @brief index operator
     * @param index the index to access
     * @return T& value by reference at(index);
     */
    JUMP_INTEROPABLE
    char_t& operator[](const std::size_t& index) {
        return at(index);
    }

    /**
     * @brief index operator
     * @param index the index to access
     * @return T& value by const reference at(index);
     */
    JUMP_INTEROPABLE
    char_t& operator[](const std::size_t& index) const {
        return at(index);
    }

    /**
     * @brief transfer all data to device
     */
    void to_device() {
        buffer_.to_device();
    }

    /**
     * @brief transfer all data from device
     */
    void from_device() {
        buffer_.from_device();
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
     * @brief access the primary location of this string
     * @return the main location memory is allocated on (if HOST, may also have a DEVICE segment)
     */
    memory_t location() const {
        return buffer_.location();
    }

private:
    /**
     * @brief internal allocation function
     * @param str string to allocate for
     * @param location the location to allocate to
     */
    void allocate(const char* str, const memory_t& location) {
        allocate(str, string_utils::length(str), location);
    }

    /**
     * @brief internal allocation function
     * @param str string to allocate for
     * @param length  length of the string to allocate
     * @param location the location to allocate to
     */
    void allocate(const char* str, const std::size_t& length, const memory_t& location) {
        if(location == memory_t::DEVICE)
            throw std::runtime_error("Unable to allocate jump::string directly to device");
        buffer_.allocate(length + 1, location);
    }

    /**
     * @brief copies a string into the buffer (assumes pre-allocation)
     * @param str string to copy into buffer
     */
    void copy_string(const char* str) {
        char* target_buffer = data();

        for(std::size_t index = 0; true; ++index) {
            target_buffer[index] = str[index];
            if(str[index] == 0)
                break;
        }
    }

    //! Internal memory buffer that stores the string
    memory_buffer buffer_;

}; /* class basic_string */


//! Define a string view type to use (template basic_string_view to char)
using string = basic_string<char>;

} /* namespace jump */

#endif /* JUMP_STRING_HPP_ */
