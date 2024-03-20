/**
 * @file string_view.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief a constexpr string_view implementation to allow
 *  string operations on GPU with --expt-relaxed-constexpr.
 *  (WARNING: NOT ALL OPERATIONS ARE IMPLEMENTED, JUST BARE MINIMUM
 *  FOR WHAT I NEED.)
 * @date 2024-02-07
 */
#ifndef JUMP_STRING_VIEW_HPP_
#define JUMP_STRING_VIEW_HPP_

// JUMP
#include <jump/device_interface.hpp>
#include <jump/string_utils.hpp>
#include <jump/string.hpp>

// STD
#include <string>

namespace jump {

/**
 * @brief constexpr string_view implementation for GPU
 *  constexpr string operations. Is not a complete implementation,
 *  just the minimum implementation for what I need.
 *
 * @note since the string_view is a weak pointer to a string
 *  in memory it does not have awareness of device or ability
 *  to transfer between host / device. As such it is only valid
 *  on the device on which it was constructed. (unless constructed
 *  from a unified memory string by dumb luck), much like a virtual
 *  class it cannot cross the CPU / GPU boundary safely.
 *
 * @tparam char_t The character type to use
 */
template<typename char_t = char>
class basic_string_view {
public:
    //! Defines the value_type that is used
    using value_type = char_t;

    /**
     * @brief constructs a string view from a const char*
     */
    JUMP_INTEROPABLE
    constexpr basic_string_view(const char_t* data) noexcept :
        data_(data),
        len_(string_utils::length(data))
    {}

    /**
     * @brief constructs a string view from a const string
     */
    JUMP_INTEROPABLE
    constexpr basic_string_view(const string& data) noexcept :
        data_(data.c_str()),
        len_(data.len())
    {}

    /**
     * @brief constructs a string view from a const std::string
     */
    constexpr basic_string_view(const std::string& data) noexcept :
        data_(data.c_str()),
        len_(data.size())
    {}

    //! Default constructor - nullify it all!
    JUMP_INTEROPABLE
    constexpr basic_string_view() noexcept : data_(nullptr), len_(0) {}
    //! Default copy constructor
    JUMP_INTEROPABLE
    constexpr basic_string_view(const basic_string_view&) noexcept = default;

    /**
     * @brief gets the length of this string (same as size())
     * @return the length
     */
    JUMP_INTEROPABLE
    constexpr std::size_t len() const noexcept {
        return len_;
    }

    /**
     * @brief gets the length of the string (same as len())
     * @return the length
     */
    JUMP_INTEROPABLE
    constexpr std::size_t size() const noexcept {
        return len_;
    }

    /**
     * @brief gets a pointer to the raw data
     * @return the string data (null-terminated)
     */
    JUMP_INTEROPABLE
    constexpr const char_t* data() const noexcept {
        return data_;
    }

    /**
     * @brief operator for indexing into the string view
     * @param idx index to get
     * @return constexpr char the character at that index
     * @note if idx >= len(), then it just returns the last index in the string
     */
    JUMP_INTEROPABLE
    constexpr char operator[](std::size_t idx) const noexcept {
        if (idx < len_)
            return data_[idx];
        return data_[len() - 1];
    }

    /**
     * @brief operator for comparison of strings
     * @param a string_view to compare
     * @param b string_view to compare
     * @return true if a and b are the same, false if not
     */
    JUMP_INTEROPABLE
    friend constexpr bool operator==(const basic_string_view& a, const basic_string_view& b) noexcept {
        return string_utils::compare(a, b) == 0;
    }

    /**
     * @brief operator for neq comparison of strings
     * @param a string_view to compare
     * @param b string_view to compare
     * @return true if a and b are different, false if not
     */
    JUMP_INTEROPABLE
    friend constexpr bool operator!=(const basic_string_view& a, const basic_string_view& b) noexcept {
        return string_utils::compare(a, b) != 0;
    }

private:
    //! String data that we are pointing to
    const char_t* data_;
    //! Length of the string we are pointing to
    const std::size_t len_;

}; /* class basic_string_view */

//! Define a string view type to use (template basic_string_view to char)
using string_view = basic_string_view<char>;

} /* namespace jump */

#endif /* JUMP_STRING_VIEW_HPP_ */
