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

/**
 * @brief container class that is able to hold multi-dimensional
 *  arrays that are fixed after creation of the array
 * @tparam T the type of object that is being held
 * @tparam _max_dims the maximum number of dimensions for this container
 */
template<typename T, std::size_t _max_dims = 4>
class multi_array {
public:
    struct axes {
        JUMP_INTEROPABLE
        axes() {
            for(std::size_t i = 0; i < _max_dims; ++i) {
                axes_[i] = true;
            }
        }

        JUMP_INTEROPABLE
        axes(const std::initializer_list<int>& axes_to_mark) {
            for(std::size_t i = 0; i < _max_dims; ++i) {
                axes_[i] = false;
            }
            for(const auto& axe_to_mark : axes_to_mark) {
                if(axe_to_mark >= _max_dims) continue;
                axes_[axe_to_mark] = true;
            }
        }

        /**
         * @brief construct axes with specific axes values
         * @tparam IndexT the type used for an specific dimension
         * @param val force there to be at least one std::size_t value to use this constructor
         * @param vals index values
         * @note makes sure that the number of arguments is not greater than _max_dims
         */
        template<typename... DimT>
        JUMP_INTEROPABLE
        axes(const std::size_t& dim, const DimT&... dims) {
            static_assert(multi_array_helpers::can_be_size_t<DimT...>(), "all DimT must be castable to std::size_t");
            static_assert(sizeof...(DimT) + 1 <= _max_dims, "Number of indexes must be less than _max_dims");
            // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
            if constexpr(multi_array_helpers::can_be_size_t<DimT...>()) {
                for(std::size_t i = 0; i < _max_dims; ++i) {
                    axes_[i] = false;
                }
                #if JUMP_ON_DEVICE
                    assert(dim < _max_dims && "dim must be less than _max_dims");
                #else
                    if(dim >= _max_dims)
                        throw std::out_of_range("dim " + std::to_string(dim) + " >= _max_dims " + std::to_string(_max_dims));
                #endif
                axes_[dim] = true;
                std::size_t dim = 1;
                for(const auto p : {dims...}) {
                    auto current_dim = static_cast<std::size_t>(p);
                    #if JUMP_ON_DEVICE
                        assert(current_dim < _max_dims && "dim must be less than _max_dims");
                    #else
                        if(current_dim >= _max_dims)
                            throw std::out_of_range("dim " + std::to_string(current_dim) + " >= _max_dims " + std::to_string(_max_dims));
                    #endif
                    axes_[current_dim] = true;
                }
            }
        }

        /**
         * @brief index into the dimensions and get the axis value
         * @param dim the dimension to get the axis selection for
         * @return bool by reference
         */
        JUMP_INTEROPABLE
        bool& operator[](const std::size_t& dim) {
            #if JUMP_ON_DEVICE
                assert(dim < _max_dims && "dim must be less than _max_dims");
            #else
                if(dim >= _max_dims)
                    throw std::out_of_range("dim " + std::to_string(dim) + " must be less than _max_dims " + std::to_string(_max_dims));
            #endif
            return axes_[dim];
        }

        /**
         * @brief index into the dimensions and get the axis value
         * @param dim the dimension to get the axis selection for
         * @return bool by const-reference
         */
        JUMP_INTEROPABLE
        const bool& operator[](const std::size_t& dim) const {
            #if JUMP_ON_DEVICE
                assert(dim < _max_dims && "dim must be less than _max_dims");
            #else
                if(dim >= _max_dims)
                    throw std::out_of_range("dim " + std::to_string(dim) + " must be less than _max_dims " + std::to_string(_max_dims));
            #endif
            return axes_[dim];
        }

        bool axes_[_max_dims];
    };

    /**
     * @brief a class that can represent multi-dimensional indices
     * @note currently the indices have no concept of the actual number
     *  of dimensions that are used.
     */
    struct indices {
        /**
         * @brief default constructor for indices - 0 out each index
         */
        JUMP_INTEROPABLE
        indices() {
            for(std::size_t i = 0; i < _max_dims; ++i)
                indices_[i] = 0;
            dim_count_ = _max_dims;
        }

        /**
         * @brief construct indices with values
         * @tparam IndexT the type used for an index value
         * @param val force there to be at least one std::size_t value to use this constructor
         * @param vals index values
         * @note makes sure that the number of arguments is not greater than _max_dims
         */
        template<typename... IndexT>
        JUMP_INTEROPABLE
        indices(const std::size_t& val, const IndexT&... vals) {
            static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
            static_assert(sizeof...(IndexT) + 1 <= _max_dims, "Number of indexes must be less than _max_dims");
            // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
            if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
                indices_[0] = val;
                std::size_t dim = 1;
                for(const auto p : {vals...}) {
                    indices_[dim++] = static_cast<std::size_t>(p);
                }
                for(std::size_t i = dim; i < _max_dims; ++i)
                    indices_[i] = 0;
            }
        }

        /**
         * @brief construct indices from an array of size values
         * @param sizes the size values to construct from
         */
        JUMP_INTEROPABLE
        indices(const std::size_t* sizes, const std::size_t& dim_count = _max_dims) {
            for(std::size_t i = 0; i < dim_count; ++i)
                indices_[i] = sizes[i];
        }

        /**
         * @brief index into the dimensions and get the index there
         * @param dim the dimension to get the index for
         * @return index value by reference
         */
        JUMP_INTEROPABLE
        std::size_t& operator[](const std::size_t& dim) {
            #if JUMP_ON_DEVICE
                assert(dim < dim_count_ && "dim must be less than dim_count_");
            #else
                if(dim >= dim_count_)
                    throw std::out_of_range("dim " + std::to_string(dim) + " must be less than dim_count_ " + std::to_string(dim_count_));
            #endif
            return indices_[dim];
        }
    
        /**
         * @brief index into the dimensions and get the index there
         * @param dim the dimension to get the index for
         * @return index value by const reference
         */
        JUMP_INTEROPABLE
        const std::size_t& operator[](const std::size_t& dim) const {
            #if JUMP_ON_DEVICE
                assert(dim < dim_count_ && "dim must be less than dim_count_");
            #else
                if(dim >= dim_count_)
                    throw std::out_of_range("dim " + std::to_string(dim) + " must be less than dim_count_ " + std::to_string(dim_count_));
            #endif
            return indices_[dim];
        }

        JUMP_INTEROPABLE
        const std::size_t& dims() const {
            return dim_count_;
        }

        JUMP_INTEROPABLE
        std::size_t& dims() {
            return dim_count_;
        }

        //! Stores the index values
        std::size_t indices_[_max_dims];
        //! Stores the number of dimensions
        std::size_t dim_count_;

    }; /* struct indices */

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
        size_(arr.size_)
    {}

    /**
     * @brief move-construct a new multi_array
     * @param arr the array to move from
     */
    multi_array(multi_array&& arr):
        buffer_(std::move(arr.buffer_)),
        size_(arr.size_)
    {}

    /**
     * @brief copy-assignment to this multi_array
     * @param arr the array to copy from
     * @return multi_array& this array which now shares ownership of data with arr
     */
    multi_array& operator=(const multi_array& arr) {
        dereference_buffer();
        buffer_ = arr.buffer_;
        size_ = arr.size_;
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
        size_ = arr.size_;
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
    std::size_t dims() const {
        return size_.dims();
    }

    /**
     * @brief access the shape with an indices representation
     * @return indices containg the sizes of all dimensions
     */
    indices shape() const {
        return size_;
    }

    /**
     * @brief gets the shape of the array (size of a given dimension)
     * @param dim the index of the dimension to get the size for,
     *  must be <= dims();
     * @return the size of dimension dim
     */
    JUMP_INTEROPABLE
    std::size_t shape(const std::size_t& dim) const {
        #if JUMP_ON_DEVICE
            assert(dim < size_.dims());
        #else
            if (dim >= size_.dims())
                throw std::out_of_range("dim " + std::to_string(dim) + " >= dims " + std::to_string(size_.dims()));
        #endif
        return size_[dim];
    }

    /**
     * @brief gets the size (total number of all elements contained) of the multi_array
     * @return the size 
     */
    JUMP_INTEROPABLE
    std::size_t size() const {
        std::size_t result = 1;
        for(std::size_t i = 0; i < size_.dims(); ++i) {
            result *= size_[i];
        }
        return result;
    }

    /**
     * @brief access an element
     * @tparam IndexT the type used to express index
     * @param vals index values
     * @return the element in the multi_array at indices by reference
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    T& at(const IndexT&... vals) const {
        static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
            const std::size_t n = sizeof...(IndexT);
            #if JUMP_ON_DEVICE
                assert(n <= size_.dims() && "at() operator is requesting a dimension greater than is possible");
            #else
                if(n > size_.dims())
                    throw std::out_of_range("requested dimensions " + std::to_string(n) + " >= dims " + std::to_string(size_.dims()));
            #endif

            return at(indices(vals...));
        }
    }

    /**
     * @brief access an element
     * @param vals the index values
     * @return the element in the multi_array at the indices (by reference)
     */
    JUMP_INTEROPABLE
    T& at(const indices& vals) const {
        for(std::size_t current_dim = 0; current_dim < size_.dims(); ++current_dim) {
            #if JUMP_ON_DEVICE
                assert(vals[current_dim] < size_[current_dim] && "at() operator is requesting an index greater than is possible");
            #else
                if(vals[current_dim] >= size_[current_dim])
                    throw std::out_of_range("index for dim " + std::to_string(current_dim) + ", " + std::to_string(vals[current_dim]) + " >= " + std::to_string(size_[current_dim]));
            #endif
        }

        auto index = indices_to_index(vals);
        return buffer_.data<T>()[index];
    }

    /**
     * @brief transfer all data to device
     */
    void to_device() {
        buffer_.to_device();
    }

    /**
     * @brief transfer all data from device to the host
     */
    void from_device() {
        buffer_.from_device();
    }

    /**
     * @brief sync device pointers before usage on device
     */
    void sync() {
        buffer_.sync();
    }

    /**
     * @brief directly access the memory buffer containing all the underlying data
     * @return the memory buffer by const reference
     */
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
        if(dimensions.size() > _max_dims) {
            throw std::out_of_range("Requested dims " + std::to_string(dimensions.size()) + " is greater than the container max_dims " + std::to_string(_max_dims));
        }
        size_.dims() = dimensions.size();

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
     * @param index_vals the indices into the array
     * @return std::size_t the index into the buffer the indices represent
     * @note this has no range checking - if we are performing operations
     *  where range is already guaranteed to be safe then we can save
     *  some computation using this instead of something like at()
     */
    JUMP_INTEROPABLE
    std::size_t indices_to_index(const indices& index_vals) const {
        std::size_t index = 0;
        for(std::size_t current_dim = 0; current_dim < dims(); ++current_dim) {
            std::size_t multiplier = 1;
            for(std::size_t i = current_dim + 1; i < dims(); ++i) {
                multiplier *= size_[i];
            }
            index += index_vals[current_dim] * multiplier;
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
    indices size_;
    // std::size_t size_[_max_dims];
    // //! Track the actual number of dimensions (must be <= _max_dims)
    // std::size_t dims_;
    //! The buffer containing all contained object data
    memory_buffer buffer_;

}; /* class multi_array */

} /* namespace jump */

#endif /* JUMP_MULTI_ARRAY_HPP_ */