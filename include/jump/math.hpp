#ifndef JUMP_MATH_HPP_
#define JUMP_MATH_HPP_

#include <jump/device_interface.hpp>

#include <cmath>

namespace jump {

//! Round operator
template<typename T>
JUMP_INTEROPABLE
T round(const T& v) {
    #if JUMP_ON_DEVICE
        return ::round(v);
    #else
        return std::round(v);
    #endif
}

//! Floor operator
template<typename T>
JUMP_INTEROPABLE
T floor(const T& v) {
    #if JUMP_ON_DEVICE
        return ::floor(v);
    #else
        return std::floor(v);
    #endif
}

//! Min operator
template<typename T>
JUMP_INTEROPABLE
T min(const T& a, const T& b) {
    #if JUMP_ON_DEVICE
        return ::min(a, b);
    #else
        return std::min(a, b);
    #endif
}

//! isnan operator
template<typename T>
JUMP_INTEROPABLE
bool isnan(const T& a) {
    #if JUMP_ON_DEVICE
        return ::isnan(a);
    #else
        return std::isnan(a);
    #endif
}

//! Exp operator
template<typename T>
JUMP_INTEROPABLE
bool exp(const T& a) {
    #if JUMP_ON_DEVICE
        return ::exp(a);
    #else
        return std::exp(a);
    #endif
}

} /* namespace jump */

#endif /* JUMP_MATH_HPP_ */
