/**
 * @file string_utils.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief General string-utils for use in string_view.hpp and string.hpp
 * @date 2024-02-06
 */
#ifndef JUMP_STRING_UTILS_HPP_
#define JUMP_STRING_UTILS_HPP_

#include <memory>

namespace jump {

namespace string_utils {

//! Thanks https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ return "Error during formatting."; }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


/**
 * @brief template function for computing the length of a string
 * @param data the string data we are checking (some class type with len() defined)
 * @tparam string_t the string type to use
 * @return the length of the string
 */
template<typename string_t>
constexpr static std::size_t length(const string_t& data) noexcept {
    return data.len();
}

/**
 * @brief string length function specialized for a character array
 * @param data the string data we are checking the length of (null-terminated)
 * @return the length of the string (how many characters before termination)
 */
constexpr static std::size_t length(const char*& data) noexcept {
    if(!data)
        return 0;

    std::size_t len = 0;
    while(data[len]) ++len;
    return len;
}

/**
 * @brief comparison operator for any string type
 * @param a one string-type to compare
 * @param b the other string-type to compare
 * @tparam string_t the string type to use (must have data() and len() defined)
 * @return -1 if a is alphabetically less than or length-wise less than b, 1 if more that, 0 if they are the same
 */
template<typename string_t>
constexpr int compare(const string_t& a, const string_t& b) {
    if (a.len() < b.len()) {
        return -1;
    }
    if (a.len() > b.len()) {
        return 1;
    }

    const std::size_t cmp_len = a.len() < b.len() ? a.len() : b.len();
    const auto a_data = a.data();
    const auto b_data = b.data();
    for(std::size_t i = 0; i < cmp_len; ++i) {
        if(a_data[i] < b_data[i]) {
            return -1;
        }
        if(a_data[i] > b_data[i]) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief comparison operator specialized for character array
 * @param a one character array to compare against
 * @param b the other character array to compare against
 * @return -1 if a is alphabetically less than or length-wise less than b, 1 if more that, 0 if they are the same
 */
constexpr int compare(const char*& a, const char*& b) {
    const auto a_len = length(a);
    const auto b_len = length(b);

    if (a_len < b_len) {
        return -1;
    }
    if (a_len > b_len) {
        return 1;
    }
    
    const std::size_t cmp_len = a_len < b_len ? a_len : b_len;
    for(std::size_t i = 0; i < cmp_len; ++i) {
        if(a[i] < b[i]) {
            return -1;
        }
        if(a[i] > b[i]) {
            return 1;
        }
    }

    return 0;
}

} /* namespace string_utils */
} /* namespace jump */

#endif
