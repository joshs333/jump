#ifndef JUMP_MULTI_ARRAY_HPP_
#define JUMP_MULTI_ARRAY_HPP_

// JUMP
#include <jump/memory_buffer.hpp>

namespace jump {

namespace multi_array_helpers {

//! Evaluate if a type can be cast to std::size_t (false evaluator)
template<typename, typename O = void>
struct can_be_size_t_eval : std::false_type {};

//! Evaluate if a type can be cast to std::size_t (true evaluator)
template<typename T>
struct can_be_size_t_eval<T, std::void_t<decltype(static_cast<std::size_t>(std::declval<T&>()))> > : std::true_type {};

//! Evaluate if a series of types can all be cast as std::size_t
//! Returns true if all types can be safely cast, false if not
template<typename Type, typename... Types>
constexpr bool can_be_size_t() {
    if constexpr(sizeof...(Types) > 0)
        return can_be_size_t_eval<Type>::value && can_be_size_t<Types...>();
    else
        return can_be_size_t_eval<Type>::value;
}

} /* namespace multi_array_helpers */

template<typename T, std::size_t _max_dims = 4>
class multi_array {
public:
    /**
     * @brief construct a new multi_array with the default initializer 
     *  for contained objects
     * @param dimensions the dimensions of the multi-array
     * @param location memory second to allocate
     */
    multi_array(
        const std::initializer_list<std::size_t>& dimensions,
        const memory_t& location = memory_t::HOST
    ) {
        allocate(dimensions, location);


        for(auto i = 0; i < size(); ++i)
            new(&buffer_.data<T>()[i]) T();
    }
    /**
     * @brief construct a new multi_array initializing members from a
     *  default value as specified
     * @param dimensions the dimensions of the multi-array
     * @param default_value the default value to intialize members from
     * @param location memory second to allocate
     */
    multi_array(
        const std::initializer_list<std::size_t>& dimensions,
        const T& default_value,
        const memory_t& location = memory_t::HOST
    ) {
        allocate(dimensions, location);

        for(auto i = 0; i < size(); ++i)
            buffer_.data<T>()[i] = default_value;
    }

    /**
     * @brief copy-construct a new multi_array
     * @param arr the array to copy from
     * @note after this operation the new multi_array and arr
     *  will share ownership of the underlying data
     */
    multi_array(const multi_array& arr):
        buffer_(arr.buffer_),
        dims_(arr.dims_)
    {
        for(int i = 0; i < dims_; ++i)
            size_[i] = arr.size_[i];
    }

    /**
     * @brief move-construct a new multi_array
     * @param arr the array to move from
     */
    multi_array(multi_array&& arr):
        buffer_(std::move(arr.buffer_)),
        dims_(arr.dims_)
    {
        for(int i = 0; i < dims_; ++i)
            size_[i] = arr.size_[i];
    }

    /**
     * @brief copy-assignment to this multi_array
     * @param arr the array to copy from
     * @return multi_array& this array which now shares ownership of data with arr
     */
    multi_array& operator=(const multi_array& arr) {
        dereference_buffer();
        buffer_ = arr.buffer_;
        dims_ = arr.dims_;
        for(int i = 0; i < dims_; ++i)
            size_[i] = arr.size_[i];
        return *this;
    }

    /**
     * @brief move-assignment to this multi_array
     * @param arr the array to copy from
     * @return multi_array& this array which has taken ownership from arr
     */
    multi_array& operator=(multi_array&& arr) {
        dereference_buffer();
        buffer_ = std::move(arr.buffer_);
        dims_ = arr.dims_;
        for(int i = 0; i < dims_; ++i)
            size_[i] = arr.size_[i];
        return *this;
    }

    /**
     * @brief destroy the object!
     */
    ~multi_array() {
        dereference_buffer();
    }
    
    /**
     * @brief gets the number of dimensions of this multi_array
     * @return std::size_t the number of dimensions 
     */
    JUMP_INTEROPABLE
    std::size_t dims() {
        return dims_;
    }

    /**
     * @brief gets the shape of the array (size of a given dimension)
     * @param dim the index of the dimension to get the size for,
     *  must be <= dims();
     * @return the size of dimension dim
     */
    JUMP_INTEROPABLE
    std::size_t shape(const std::size_t& dim) {
        #if JUMP_ON_DEVICE
            assert(dim < dims_);
        #else
            if (dim >= dims_)
                throw std::out_of_range("dim " + std::to_string(dim) + " >= dims " + std::to_string(dims_));
        #endif
        return size_[dim];
    }

    /**
     * @brief gets the size (total number of all elements contained) of the multi_array
     * @return the size 
     */
    JUMP_INTEROPABLE
    std::size_t size() {
        std::size_t result = 1;
        for(std::size_t i = 0; i < dims_; ++i) {
            result *= size_[i];
        }
        return result;
    }

    /**
     * @brief access an element
     * @tparam IndexT the type used to express index
     * @param indices some series of indices to identify which element to access
     * @return the element in the multi_array at indices by reference
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    T& at(const IndexT&... indices) {
        static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
            std::size_t current_dim = 0;
            const std::size_t n = sizeof...(IndexT);
            #if JUMP_ON_DEVICE
                assert(n <= dims_ && "at() operator is requesting a dimension greater than is possible");
            #else
                if(n > dims_)
                    throw std::out_of_range("requested dimensions " + std::to_string(n) + " >= dims " + std::to_string(dims_));
            #endif

            for(const auto p : {indices...}) {
                std::size_t idx = static_cast<std::size_t>(p);
                #if JUMP_ON_DEVICE
                    assert(idx < size_[current_dim] && "at() operator is requesting an index greater than is possible");
                #else
                    if(idx >= size_[current_dim])
                        throw std::out_of_range("index for dim " + std::to_string(current_dim) + ", " + std::to_string(idx) + " >= " + std::to_string(size_[current_dim]));
                #endif
                ++current_dim;
            }
            auto index = indices_to_index(indices...);
            return buffer_.data<T>()[index];
        }
    }

    void to_device() {
        buffer_.to_device();
    }

    void from_device() {
        buffer_.from_device();
    }

    void sync() {
        buffer_.sync();
    }

    const memory_buffer& buffer() const {
        return buffer_;
    }

private:
    /**
     * @brief an internal method shared between constructors to perform basic allocation
     * @param dimensions the dimensions to set
     * @param location the memory locatin to allocate
     */
    void allocate(const std::vector<std::size_t>& dimensions, const memory_t& location) {
        dims_ = dimensions.size();

        if(dims_ > _max_dims) {
            throw std::out_of_range("Requested dims " + std::to_string(dims_) + " is greater than the container max_dims " + std::to_string(_max_dims));
        }

        auto total_size = 1;
        auto idx = 0;
        for(auto& s : dimensions) {
            total_size *= s;
            size_[idx] = s;
            ++idx;
        }

        buffer_.allocate<T>(total_size, location);
    }

    /**
     * @brief performs a conversion from indices to index in the buffer
     * @tparam IndexT the type used for indices
     * @param indices the indices into the array
     * @return std::size_t the index into the buffer the indices represent
     * @note this has no range checking - if we are performing operations
     *  where range is already guaranteed to be safe then we can save
     *  some computation using this instead of something like at()
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    std::size_t indices_to_index(const IndexT&... indices) {
        std::size_t current_dim = 0;
        std::size_t index = 0;
        for(const auto& p : {indices...}) {
            std::size_t idx = static_cast<std::size_t>(p);
            std::size_t multiplier = 1;
            for(std::size_t i = current_dim + 1; i < dims_; ++i) {
                multiplier *= size_[i];
            }
            index += idx * multiplier;
            ++current_dim;
        }
        return index;
    }

    /**
     * @brief manually perform buffer dereferencing
     *  to ensure descruction happens properly
     */
    void dereference_buffer() {
        buffer_.release([&](){
            for(std::size_t i = 0; i < size(); ++i) {
                std::exception e;
                auto ex = false;
                try {
                    buffer_.data<T>()[i].~T();
                } catch(std::exception& er) {
                    e = er;
                }
                if(ex)
                    throw e;
            }
        });
    }

    //! Track the size of the array
    std::size_t size_[_max_dims];
    //! Track the actual number of dimensions (must be <= _max_dims)
    std::size_t dims_;
    //! The buffer containing all contained object data
    memory_buffer buffer_;

}; /* class multi_array */

} /* namespace jump */

#endif /* JUMP_MULTI_ARRAY_HPP_ */