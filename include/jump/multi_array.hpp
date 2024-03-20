/**
 * @file multi_array.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief A multi-dimensional array over some memory buffer.
 * @date 2023-04-25
 */
#ifndef JUMP_MULTI_ARRAY_HPP_
#define JUMP_MULTI_ARRAY_HPP_

// JUMP
#include <jump/memory_buffer.hpp>

// STD
#include <chrono>
#include <limits>

namespace jump {

//! helpers for the multi_array
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
    return false;
}

} /* namespace multi_array_helpers */

/**
 * @brief could be used to note which axes to loop over
 *  for foreach calls.
 * @tparam _max_dims the maximum number of dimensions to support
 */
template<std::size_t _max_dims = 4>
struct axes {
    /**
     * @brief get the max dims for these axes
     * @return constexpr std::size_t the max dims
     */
    constexpr std::size_t max_dims() const {
        return _max_dims;
    }

    /**
     * @brief construct axes (all true)
     */
    JUMP_INTEROPABLE
    axes() {
        for(std::size_t i = 0; i < _max_dims; ++i) {
            axes_[i] = true;
        }
    }

    /**
     * @brief construct axes with some subset of the axes
     * @param axes_to_mark a list of the axes to mark
     */
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
     * @param dim force there to be at least one std::size_t value to use this constructor
     * @param dims dims to select
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

    //! The actual data of what axes are selected
    bool axes_[_max_dims];

}; /* struct axes */


/**
 * @brief a class that can represent multi-dimensional multi_indices
 */
template<std::size_t _max_dims = 4>
struct multi_indices {
    /**
     * @brief construct indices with some number of dimensions all set to zero
     * @param size the number of dimensions
     */
    JUMP_INTEROPABLE
    static multi_indices zero(const std::size_t& size = _max_dims) {
        multi_indices result;
        for(std::size_t i = 0; i < size; ++i) {
            result[i] = 0;
        }
        result.dim_count_ = size;
        return result;
    }

    /**
     * @brief default constructor for multi_indices - 0 out each index
     */
    JUMP_INTEROPABLE
    multi_indices() {
        for(std::size_t i = 0; i < _max_dims; ++i)
            multi_indices_[i] = 0;
        dim_count_ = _max_dims;
    }

    /**
     * @brief construct multi_indices with values
     * @tparam IndexT the type used for an index value
     * @param val force there to be at least one std::size_t value to use this constructor
     * @param vals index values
     * @note makes sure that the number of arguments is not greater than _max_dims
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    multi_indices(const std::size_t& val, const IndexT&... vals) {
        static_assert(sizeof...(IndexT) + 1 <= _max_dims, "Number of indexes must be less than _max_dims");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(sizeof...(IndexT) == 0) {
            dim_count_ = 1;
            multi_indices_[0] = val;
            for(std::size_t i = 1; i < _max_dims; ++i)
                multi_indices_[i] = 0;
        } else {
            static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
            if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
                dim_count_ = sizeof...(IndexT) + 1;
                multi_indices_[0] = val;
                std::size_t dim = 1;
                for(const auto p : {static_cast<std::size_t>(vals)...}) {
                    multi_indices_[dim++] = static_cast<std::size_t>(p);
                }
                for(std::size_t i = dim; i < _max_dims; ++i)
                    multi_indices_[i] = 0;
            }
        }
    }

    /**
     * @brief construct multi_indices from an array of size values
     * @param sizes the size values to construct from
     * @param dim_count the number of dimensions of these indices (default = _max_dims)
     */
    JUMP_INTEROPABLE
    multi_indices(const std::size_t* sizes, const std::size_t& dim_count = _max_dims) {
        for(std::size_t i = 0; i < dim_count; ++i)
            multi_indices_[i] = sizes[i];
    }

    /**
     * @brief index into the dimensions and get the index there
     * @param dim the dimension to get the index for
     * @return index value by reference
     */
    JUMP_INTEROPABLE
    std::size_t& operator[](std::size_t dim) {
        #if JUMP_ON_DEVICE
            assert(dim < dim_count_ && "dim must be less than dim_count_");
        #else
            if(dim >= dim_count_)
                throw std::out_of_range("dim " + std::to_string(dim) + " must be less than dim_count_ " + std::to_string(dim_count_));
        #endif
        return multi_indices_[dim];
    }

    /**
     * @brief index into the dimensions and get the index there
     * @param dim the dimension to get the index for
     * @return index value by const reference
     */
    JUMP_INTEROPABLE
    const std::size_t& operator[](std::size_t dim) const {
        #if JUMP_ON_DEVICE
            assert(dim < dim_count_ && "dim must be less than dim_count_");
        #else
            if(dim >= dim_count_)
                throw std::out_of_range("dim " + std::to_string(dim) + " must be less than dim_count_ " + std::to_string(dim_count_));
        #endif
        return multi_indices_[dim];
    }

    /**
     * @brief get the dimension count (const ref)
     * @return the number of dimensions
     */
    JUMP_INTEROPABLE
    const std::size_t& dims() const {
        return dim_count_;
    }

    /**
     * @brief get the dimension count (const ref)
     * @return the number of dimensions
     */
    JUMP_INTEROPABLE
    std::size_t& dims() {
        return dim_count_;
    }

    /**
     * @brief gets the `volume` of a hyper-rectangle with dimensions / size of these indices
     * @return the offset i guess...
     * @todo rename maybe? lol this name sense to me
     *  5 hours ago but for some reason no longer does
     */
    JUMP_INTEROPABLE
    std::size_t offset() const {
        std::size_t result = 1;
        for(std::size_t i = 0 ; i < dims(); ++i) {
            result *= multi_indices_[i];
        }
        return result;
    }

    /**
     * @brief pre-fix incrementer
     * @return this object
     */
    JUMP_INTEROPABLE
    multi_indices& operator++() {
        if(dim_count_ == 0) return *this;
        for(std::size_t i = 1; i <= dim_count_; ++i) {
            // if the index is maxed, we rollover
            if(multi_indices_[dim_count_ - i] == std::numeric_limits<std::size_t>::max()) {
                multi_indices_[dim_count_ - i] = 0;
            } else {
                multi_indices_[dim_count_ - i]++;
                break;
            }
        }
        return *this;
    }

    /**
     * @brief modulo by another indices, useful indices
     *  perforing a modulo over indices representing the size of a multi-array
     * @param other the indices to modulo against
     * @return modulo'd indices
     */
    JUMP_INTEROPABLE
    multi_indices operator%(const multi_indices& other) {
        auto result = *this;
        result %= other;
        return result;
    }

    /**
     * @brief modulo by another indices, useful indices
     *  perforing a modulo over indices representing the size of a multi-array
     * @param other the indices to modulo against
     * @return this class modulo'd
     * @note I used to pass other by const-reference but index operators would not evaluate
     *  properly. I'm either crazy, missing something obvious (or insidious) or just maybe there's a compiler bug.
     */
    JUMP_INTEROPABLE
    multi_indices& operator%=(multi_indices other) {
        if(dim_count_ == 0 || dim_count_ == 1) return *this;
        for(std::size_t i = 1; i < dim_count_; ++i) {
            if (multi_indices_[dim_count_ - i] >= other.multi_indices_[dim_count_ - i]) {
                if(i < dim_count_) {
                    multi_indices_[dim_count_ - i - 1] += multi_indices_[dim_count_ - i] / other.multi_indices_[dim_count_ - i];
                }
                multi_indices_[dim_count_ - i] = multi_indices_[dim_count_ - i] % other.multi_indices_[dim_count_ - i];
            }
        }
        return *this;
    }

    /**
     * @brief equal comparison against another indices
     * @param other the indices to compare against
     * @return this == other
     */
    JUMP_INTEROPABLE
    bool operator==(const multi_indices& other) {
        if(other.dim_count_ != dim_count_)
            return false;
        for(std::size_t i = 0; i < dim_count_; ++i) {
            if(multi_indices_[i] != other.multi_indices_[i])
                return false;
        }
        return true;
    }

    /**
     * @brief less-than comparison against another indices
     * @param other the indices to compare against
     * @return this < other
     */
    JUMP_INTEROPABLE
    bool operator<(const multi_indices& other) {
        for(std::size_t i = 0; i < dim_count_ && i < other.dim_count_; ++i) {
            // if we are greater than the other, we definitely aren't less
            if(multi_indices_[i] > other.multi_indices_[i])
                return false;
            // if we are less than the other, we are definitely less
            if(multi_indices_[i] < other.multi_indices_[i])
                return true;
        }
        // if we go through all the indices higher order to lower and
        // haven't returned yet, all are equal and we are not less than
        return false;
    }

    /**
     * @brief less-than or equal comparison against another indices
     * @param other the indices to compare against
     * @return this <= other
     */
    JUMP_INTEROPABLE
    bool operator<=(const multi_indices& other) {
        for(std::size_t i = 0; i < dim_count_ && i < other.dim_count_; ++i) {
            // if we are greater we definitely aren't less
            if(multi_indices_[i] > other.multi_indices_[i])
                return false;
        }
        // if we get here, each index is either less or equal and we are good
        return true;
    }

    /**
     * @brief greater-than comparison against another indices
     * @param other the indices to compare against
     * @return this > other
     */
    JUMP_INTEROPABLE
    bool operator>(const multi_indices& other) {
        for(std::size_t i = 0; i < dim_count_ && i < other.dim_count_; ++i) {
            // a higher index is higher, we are higher
            if(multi_indices_[i] > other.multi_indices_[i])
                return true;
            // a higher index is lower, we are lower
            if(multi_indices_[i] < other.multi_indices_[i])
                return false;
        }
        return false;
    }

    /**
     * @brief greater-than or equal comparison against another indices
     * @param other the indices to compare against
     * @return this >= other
     */
    JUMP_INTEROPABLE
    bool operator>=(const multi_indices& other) {
        for(std::size_t i = 0; i < dim_count_ && i < other.dim_count_; ++i) {
            if(multi_indices_[i] < other.multi_indices_[i])
                return false;
        }
        return true;
    }

    /**
     * @brief not-equal comparison against another indices
     * @param other the indices to compare against
     * @return this != other
     */
    JUMP_INTEROPABLE
    bool operator!=(const multi_indices& other) {
        return !(*this == other);
    }

    //! Stores the number of dimensions
    std::size_t dim_count_;
    //! Stores the index values
    std::size_t multi_indices_[_max_dims];

}; /* struct multi_indices */

//! For convenience the indices type will default to the default dimensionality of multi_indices
using indices = multi_indices<>;

/**
 * @brief a simple helper class to handle comma
 *  assignment of values into multi_arrays
 * @tparam T the value type to be assigned
 */
template<typename T, std::size_t _max_dims = 4>
class multi_array_comma_helper {
public:
    //! Convenient type, indicies with the same _max_dims as this multi_array
    using indices = multi_indices<_max_dims>;

    //! Convenience type for the value type contained
    using value_type = T;

    /**
     * @brief construct the helper
     * @param start_data beginning of data to fill (not assignable)
     * @param end_data end of data to fill
     */
    JUMP_INTEROPABLE
    multi_array_comma_helper(T* start_data, T* end_data):
        start_data_(start_data),
        end_data_(end_data)
    {}

    /**
     * @brief assign into multi_array using comma operator
     * @param value the value to assign
     */
    JUMP_INTEROPABLE
    multi_array_comma_helper& operator,(const T& value) {
        #if JUMP_ON_DEVICE
            #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                assert(start_data_ < end_data_ && "Unable to assign into multi_array beyond it's range");
            #endif
        #else
            if(start_data_ >= end_data_)
                throw std::out_of_range("Unable to assign into multi_array beyond it's range");
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

}; /* class multi_array_comma_helper */


/**
 * @brief a weak-pointer view of data from a multi-array with the same interface
 * @tparam T the type of object that is being held
 * @tparam _max_dims the maximum number of dimensions for this container
 */
template<typename T, std::size_t _max_dims = 4>
class multi_array_view {
public:
    //! Convenient type, indicies with the same _max_dims as this multi_array
    using indices = multi_indices<_max_dims>;

    //! Convenience type for the value type contained
    using value_type = T;

    //! Construct an empty (null) multi_array_view
    JUMP_INTEROPABLE
    multi_array_view():
        data_(nullptr),
        size_(indices::zero(_max_dims))
    {
    }

    /**
     * @brief Construct a new multi array view object
     * @param data the raw buffer of data to point to
     * @param dimensions the dimensions of the buffer
     * @note while this is intended to be used in concert
     *  with the multi_array, this could be used to wrap
     *  any array, pretty cool write? haha
     */
    JUMP_INTEROPABLE
    multi_array_view(
        T* data,
        const indices& dimensions
    ):
        data_(data),
        size_(dimensions)
    {}

    /**
     * @brief gets the number of dimensions of this multi_array
     * @return std::size_t the number of dimensions 
     */
    JUMP_INTEROPABLE
    std::size_t dims() const {
        return size_.dims();
    }

    /**
     * @brief access the shape with an multi_indices representation
     * @return multi_indices containg the sizes of all dimensions
     */
    JUMP_INTEROPABLE
    multi_indices<_max_dims> shape() const {
        return size_;
    }

    /**
     * @brief get a zero index indices
     * @return multi_indices of the same type used by this multi_array with all zeros
     */
    JUMP_INTEROPABLE
    multi_indices<_max_dims> zero() const {
        auto result = multi_indices<_max_dims>();
        result.dim_count_ = size_.dims();
        return result;
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
            #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                assert(dim < size_.dims());
            #endif
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
     * @return the element in the multi_array at multi_indices by reference
     * @note BIG TODO(jspisak): handling where sizeof...(IndexT) < dims()
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    T& at(const IndexT&... vals) const {
        static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
            #if JUMP_ON_DEVICE
                #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                    assert(sizeof...(IndexT) <= size_.dims() && "at() operator is requesting a dimension greater than is possible");
                #endif
            #else
                if(sizeof...(IndexT) > size_.dims())
                    throw std::out_of_range("requested dimensions " + std::to_string(sizeof...(IndexT)) + " >= dims " + std::to_string(size_.dims()));
            #endif

            return at(multi_indices(vals...));
        }
    }

    /**
     * @brief access an element
     * @param vals the index values
     * @return the element in the multi_array at the multi_indices (by reference)
     * @note BIG TODO(jspisak): handling where vals.dims() < dims()
     */
    JUMP_INTEROPABLE
    T& at(const multi_indices<_max_dims>& vals) const {
        #if not JUMP_ON_DEVICE or JUMP_ENABLE_DEVICE_BOUNDS_CHECK
            for(std::size_t current_dim = 0; current_dim < size_.dims() && current_dim < vals.dims(); ++current_dim) {
                #if JUMP_ON_DEVICE
                    assert(vals[current_dim] < size_[current_dim] && "at() operator is requesting an index greater than is possible");
                #else
                    if(vals[current_dim] >= size_[current_dim])
                        throw std::out_of_range("index for dim " + std::to_string(current_dim) + ", " + std::to_string(vals[current_dim]) + " >= " + std::to_string(size_[current_dim]));
                #endif
            }
        #endif

        auto index = multi_indices_to_index(vals);
        return data_[index];
    }

    /**
     * @brief index into the array (ignore dimensions)
     * @param idx the index of element to get
     * @return element in array by referece
     */
    JUMP_INTEROPABLE
    T& operator[](const std::size_t& idx) const {
        #if not JUMP_ON_DEVICE or JUMP_ENABLE_DEVICE_BOUNDS_CHECK
            #if JUMP_ON_DEVICE
                assert(idx < size() && "[] operator is requesting an index greater than is possible");
            #else
                if(idx >= size())
                    throw std::out_of_range("index " + std::to_string(idx) + " >= " + std::to_string(size()));
            #endif
        #endif

        return data_[idx];
    }

    /**
     * @brief assign into multi_array_view using istream and comma operator
     * @param value the first value to assign
     * @return a comma helper that can take values via the comma operator to assign into the multi_array
     */
    JUMP_INTEROPABLE
    multi_array_comma_helper<T> operator<<(const T& value) const {
        return multi_array_comma_helper<T>(data_, data_ + size()), value;
    }

    /**
     * @brief view a portion of the array
     * @tparam IndexT the type used to express index
     * @param vals index values (should be one dimension less than multi_array dims())
     *  when dim of vals == dims(), it should be an at operation.
     * @return a view of the section of the array indexed at view 
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    multi_array_view<T, _max_dims> view(const IndexT&... vals) const {
        static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
            #if JUMP_ON_DEVICE
                #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                    assert(sizeof...(IndexT) <= size_.dims() && "view() operator is requesting a dimension greater than is possible");
                #endif
            #else
                if(sizeof...(IndexT) > size_.dims())
                    throw std::out_of_range("requested dimensions " + std::to_string(sizeof...(IndexT)) + " >= dims " + std::to_string(size_.dims()));
            #endif

            return view(multi_indices(vals...));
        }
    }

    /**
     * @brief view a portion of the array
     * @param vals index values (should be one dimension less than multi_array dims())
     *  when dim of vals == dims(), it should be an at operation.
     * @return a view of the section of the array indexed at view 
     */
    JUMP_INTEROPABLE
    multi_array_view<T, _max_dims> view(const multi_indices<_max_dims>& vals) const {
        #if not JUMP_ON_DEVICE or JUMP_ENABLE_DEVICE_BOUNDS_CHECK
            for(std::size_t current_dim = 0; current_dim < size_.dims() && current_dim < vals.dims(); ++current_dim) {
                #if JUMP_ON_DEVICE
                    assert(vals[current_dim] < size_[current_dim] && "view() operator is requesting an index greater than is possible");
                #else
                    if(vals[current_dim] >= size_[current_dim])
                        throw std::out_of_range("index for dim " + std::to_string(current_dim) + ", " + std::to_string(vals[current_dim]) + " >= " + std::to_string(size_[current_dim]));
                #endif
            }
            #if JUMP_ON_DEVICE
                assert(vals.dims() < size_.dims() && "view() operator must view less than the available dimensions");
            #else
                if(vals.dims() >= size_.dims())
                    throw std::out_of_range("view() operator must view less than the available dimensions");
            #endif
        #endif

        indices size = indices::zero(size_.dims() - vals.dims());
        for(std::size_t i = vals.dims(); i < size_.dims(); ++i)
            size[i - vals.dims()] = size_[i];

        auto index = multi_indices_to_index(vals);
        auto data_ptr = data_ + index;
        return multi_array_view(data_ptr, size);
    }

    /**
     * @brief returns a weak-ptr reference view of the array
     * @return a view of the whole multi-array
     */
    JUMP_INTEROPABLE
    multi_array_view<T, _max_dims> view() const {
        return multi_array_view(data_, size_);
    }

private:
    /**
     * @brief performs a conversion from multi_indices to index in the buffer
     * @param index_vals the multi_indices into the array
     * @return std::size_t the index into the buffer the multi_indices represent
     * @note this has no range checking - if we are performing operations
     *  where range is already guaranteed to be safe then we can save
     *  some computation using this instead of something like at()
     */
    JUMP_INTEROPABLE
    std::size_t multi_indices_to_index(const multi_indices<_max_dims>& index_vals) const {
        std::size_t index = 0;
        for(std::size_t current_dim = 0; current_dim < dims() && current_dim < index_vals.dims(); ++current_dim) {
            std::size_t multiplier = 1;
            for(std::size_t i = current_dim + 1; i < dims(); ++i) {
                multiplier *= size_[i];
            }
            index += index_vals[current_dim] * multiplier;
        }
        return index;
    }

    //! Pointer 
    T* data_;
    //! Size information for this view
    multi_indices<_max_dims> size_;

}; /* class multi_array_view */

/**
 * @brief container class that is able to hold multi-dimensional
 *  arrays that are fixed after creation of the array
 * @tparam T the type of object that is being held
 * @tparam _max_dims the maximum number of dimensions for this container
 */
template<typename T, std::size_t _max_dims = 4>
class multi_array {
public:
    //! Convenient type, indicies with the same _max_dims as this multi_array
    using indices = multi_indices<_max_dims>;

    //! Convenience type for the value type contained
    using value_type = T;

    multi_array():
        size_(indices::zero(_max_dims))
    {
    }

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

        if constexpr(!std::is_fundamental<T>::value) {
            if(location != memory_t::DEVICE) {
                for(auto i = 0; i < size(); ++i)
                    new(&buffer_.data<T>()[i]) T();
            }
        }
    }

    /**
     * @brief construct a new multi_array with the default initializer 
     *  for contained objects
     * @param dimensions the dimensions of the multi-array
     * @param location memory second to allocate
     */
    multi_array(
        const indices& dimensions,
        const memory_t& location = memory_t::HOST
    ) {
        std::vector<std::size_t> dims(dimensions.dims(), 0);
        for(auto i = 0; i < dimensions.dims(); ++i) {
            dims[i] = dimensions[i];
        }
        allocate(dims, location);

        if constexpr(!std::is_fundamental<T>::value) {
            if(location != memory_t::DEVICE) {
                for(auto i = 0; i < size(); ++i)
                    new(&buffer_.data<T>()[i]) T();
            }
        }
    }

    /**
     * @brief construct a new multi_array initializing members from 
        a default value as specified
     * @param dimensions the dimensions of the multi-array
     * @param default_value the default value to intialize members from
     * @param location memory second to allocate
     */
    multi_array(
        const indices& dimensions,
        const T& default_value,
        const memory_t& location = memory_t::HOST
    ) {
        std::vector<std::size_t> dims(dimensions.dims(), 0);
        for(auto i = 0; i < dimensions.dims(); ++i) {
            dims[i] = dimensions[i];
        }
        allocate(dims, location);

        if(location != memory_t::DEVICE)
            for(auto i = 0; i < size(); ++i)
                new(&buffer_.data<T>()[i]) T(default_value);
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

        if(location != memory_t::DEVICE)
            for(auto i = 0; i < size(); ++i)
                buffer_.data<T>()[i] = default_value;
    }

    /**
     * @brief allows direct assignment of the multi array to a buffer
     *  with a specified size
     * @param buffer the buffer to assign
     * @param size the size to assign
     */
    multi_array(
        const memory_buffer& buffer,
        const indices& size
    ):
        buffer_(buffer),
        size_(size)
    {}

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
     * @brief allows recasting the multi array to different types
     *  by changing the pointer type of the multi_array
     * @tparam O the type to change to
     * @return multi_array<O, _max_dims>
     */
    template<typename O>
    multi_array<O, _max_dims> recast() const {
        static_assert(sizeof(O) == sizeof(T), "Can only recast multiarrays to objects of the same size.");
        static_assert(std::is_fundamental<O>::value, "Can only recast multiarrays to fundamental types.");
        static_assert(std::is_fundamental<T>::value, "Can only recast multiarrays from fundamental types.");
        return multi_array<O, _max_dims>(buffer_, size_);
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
     * @brief access the shape with an multi_indices representation
     * @return multi_indices containg the sizes of all dimensions
     */
    JUMP_INTEROPABLE
    multi_indices<_max_dims> shape() const {
        return size_;
    }

    /**
     * @brief get a zero index indices
     * @return multi_indices of the same type used by this multi_array with all zeros
     */
    JUMP_INTEROPABLE
    multi_indices<_max_dims> zero() const {
        auto result = multi_indices<_max_dims>();
        result.dim_count_ = size_.dims();
        return result;
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
            #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                assert(dim < size_.dims());
            #endif
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
     * @return the element in the multi_array at multi_indices by reference
     * @note BIG TODO(jspisak): handling where sizeof...(IndexT) < dims()
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    T& at(const IndexT&... vals) const {
        static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
            #if JUMP_ON_DEVICE
                #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                    assert(sizeof...(IndexT) <= size_.dims() && "at() operator is requesting a dimension greater than is possible");
                #endif
            #else
                if(sizeof...(IndexT) > size_.dims())
                    throw std::out_of_range("requested dimensions " + std::to_string(sizeof...(IndexT)) + " >= dims " + std::to_string(size_.dims()));
            #endif

            return at(multi_indices(vals...));
        }
    }

    /**
     * @brief access an element
     * @param vals the index values
     * @return the element in the multi_array at the multi_indices (by reference)
     * @note BIG TODO(jspisak): handling where vals.dims() < dims()
     */
    JUMP_INTEROPABLE
    T& at(const multi_indices<_max_dims>& vals) const {
        #if not JUMP_ON_DEVICE or JUMP_ENABLE_DEVICE_BOUNDS_CHECK
            for(std::size_t current_dim = 0; current_dim < size_.dims() && current_dim < vals.dims(); ++current_dim) {
                #if JUMP_ON_DEVICE
                    assert(vals[current_dim] < size_[current_dim] && "at() operator is requesting an index greater than is possible");
                #else
                    if(vals[current_dim] >= size_[current_dim])
                        throw std::out_of_range("index for dim " + std::to_string(current_dim) + ", " + std::to_string(vals[current_dim]) + " >= " + std::to_string(size_[current_dim]));
                #endif
            }
        #endif

        auto index = multi_indices_to_index(vals);
        return buffer_.data<T>()[index];
    }

    /**
     * @brief index into the array (ignore dimensions)
     * @param index the index of element to get
     * @return element in array by referece
     */
    JUMP_INTEROPABLE
    T& operator[](const std::size_t& index) const {
        #if not JUMP_ON_DEVICE or JUMP_ENABLE_DEVICE_BOUNDS_CHECK
            #if JUMP_ON_DEVICE
                assert(index < size() && "[] operator is requesting an index greater than is possible");
            #else
                if(index >= size())
                    throw std::out_of_range("index " + std::to_string(index) + " >= " + std::to_string(size()));
            #endif
        #endif

        return buffer_.data<T>()[index];
    }

    /**
     * @brief assign into multi_array using istream and comma operator
     * @param value the first value to assign
     * @return a comma helper that can take values via the comma operator to assign into the multi_array
     */
    JUMP_INTEROPABLE
    multi_array_comma_helper<T> operator<<(const T& value) const {
        auto ptr = buffer_.data<T>();
        return multi_array_comma_helper<T>(ptr, ptr + size()), value;
    }

    /**
     * @brief view a portion of the array
     * @tparam IndexT the type used to express index
     * @param vals index values (should be one dimension less than multi_array dims())
     *  when dim of vals == dims(), it should be an at operation.
     * @return a view of the section of the array indexed at view 
     */
    template<typename... IndexT>
    JUMP_INTEROPABLE
    multi_array_view<T, _max_dims> view(const IndexT&... vals) const {
        static_assert(multi_array_helpers::can_be_size_t<IndexT...>(), "all IndexT must be castable to std::size_t");
        // wrapping the rest in this constexpr cleans up the error output if the static_assert fails :)
        if constexpr(multi_array_helpers::can_be_size_t<IndexT...>()) {
            #if JUMP_ON_DEVICE
                #if JUMP_ENABLE_DEVICE_BOUNDS_CHECK
                    assert(sizeof...(IndexT) <= size_.dims() && "view() operator is requesting a dimension greater than is possible");
                #endif
            #else
                const std::size_t n = sizeof...(IndexT);
                if(n > size_.dims())
                    throw std::out_of_range("requested dimensions " + std::to_string(n) + " >= dims " + std::to_string(size_.dims()));
            #endif

            return view(multi_indices(vals...));
        }
    }

    /**
     * @brief view a portion of the array
     * @param vals index values (should be one dimension less than multi_array dims())
     *  when dim of vals == dims(), it should be an at operation.
     * @return a view of the section of the array indexed at view 
     */
    JUMP_INTEROPABLE
    multi_array_view<T, _max_dims> view(const multi_indices<_max_dims>& vals) const {
        #if not JUMP_ON_DEVICE or JUMP_ENABLE_DEVICE_BOUNDS_CHECK
            for(std::size_t current_dim = 0; current_dim < size_.dims() && current_dim < vals.dims(); ++current_dim) {
                #if JUMP_ON_DEVICE
                    assert(vals[current_dim] < size_[current_dim] && "view() operator is requesting an index greater than is possible");
                #else
                    if(vals[current_dim] >= size_[current_dim])
                        throw std::out_of_range("index for dim " + std::to_string(current_dim) + ", " + std::to_string(vals[current_dim]) + " >= " + std::to_string(size_[current_dim]));
                #endif
            }
            #if JUMP_ON_DEVICE
                assert(vals.dims() < size_.dims() && "view() operator must view less than the available dimensions");
            #else
                if(vals.dims() >= size_.dims())
                    throw std::out_of_range("view() operator must view less than the available dimensions");
            #endif
        #endif

        indices size = indices::zero(size_.dims() - vals.dims());
        for(std::size_t i = vals.dims(); i < size_.dims(); ++i)
            size[i - vals.dims()] = size_[i];

        auto index = multi_indices_to_index(vals);
        auto data_ptr = buffer_.data<T>() + index;
        return multi_array_view(data_ptr, size);
    }

    /**
     * @brief returns a weak-ptr reference view of the array
     * @return a view of the whole multi-array
     */
    JUMP_INTEROPABLE
    multi_array_view<T, _max_dims> view() const {
        return multi_array_view(buffer_.data<T>(), size_);
    }

    /**
     * @brief allow implicit conversion to a multi_array_view
     * @return a view of the whole multi-array
     */
    JUMP_INTEROPABLE
    operator multi_array_view<T, _max_dims>() const {
        return view();
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

    /**
     * @brief access the primary location of this multi_array
     * @return the main location memory is allocated on (if HOST, may also have a DEVICE segment)
     */
    memory_t location() const {
        return buffer_.location();
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
     * @brief performs a conversion from multi_indices to index in the buffer
     * @param index_vals the multi_indices into the array
     * @return std::size_t the index into the buffer the multi_indices represent
     * @note this has no range checking - if we are performing operations
     *  where range is already guaranteed to be safe then we can save
     *  some computation using this instead of something like at()
     */
    JUMP_INTEROPABLE
    std::size_t multi_indices_to_index(const multi_indices<_max_dims>& index_vals) const {
        std::size_t index = 0;
        for(std::size_t current_dim = 0; current_dim < dims() && current_dim < index_vals.dims(); ++current_dim) {
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
            if constexpr(!std::is_fundamental<T>::value) {
                if(buffer_.location() != memory_t::DEVICE) {
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
                }
            }
        });
    }

    //! Track the size of the array
    multi_indices<_max_dims> size_;
    //! The buffer containing all contained object data
    memory_buffer buffer_;

}; /* class multi_array */

} /* namespace jump */

#endif /* JUMP_MULTI_ARRAY_HPP_ */
