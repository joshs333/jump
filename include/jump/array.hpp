#ifndef JUMP_ARRAY_HPP_
#define JUMP_ARRAY_HPP_

//! Just Multi-Processing namespace
namespace jump {

//! Helpers for the array class
namespace _array_helpers {

//! Helper to isolate indexing calculations
template<int dims>
struct indexing {
    using dim_t = std::size_t[dims];
    indexing(const dim_t& size):
        size_(size)
    {}

    std::size_t operator()

    dim_t size_;
}; /* struct indexing */

} /* namespace _array_helpers */


} /* namespace jump */

#endif /* JUMP_ARRAY_HPP_ */