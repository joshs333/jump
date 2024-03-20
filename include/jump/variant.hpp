/**
 * @file variant.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief Implements a rudimentary variant. Many thanks to the internet for inspiration and guidance
 *  on how to make the type_index work.
 * @date 2024-02-27
 * @note some gotchas that I need to document better:
 *  - this will pass all operations through to contained values (destructor, visitor, etc...)
 *  - since this is interopable, it will do this on device or host
 *  - we CANNOT check whether those functions are properly defined for the __device__ or __host__
 *  - we have disabled those warnings for convenience
 *  - you can make this do bad things if you don't pay attention
 *  - pay attention :) <3
 */
#ifndef JUMP_VARIANT_HPP_
#define JUMP_VARIANT_HPP_

// JUMP
#include <jump/device_interface.hpp>

// STD
#include <stdexcept>

namespace jump {

namespace _variant_helpers {

// Terminating case (we reach the end & the type isn't there)
template <typename TargetType, typename... OtherTypes>
struct type_index : std::integral_constant<int, std::numeric_limits<int>::min()> {};

//! When TargetType is the same as the first type in OtherTypes, this returns 0 (terminating case)
template <typename TargetType, typename... OtherTypes>
struct type_index<TargetType, TargetType, OtherTypes...> : std::integral_constant<int, 0> 
{};

//! When OtherType (a type in the pack) is not the same as TargetType, this adds 1 and recurses down the integral constant
template <typename TargetType, typename OtherType, typename... OtherTypes>
struct type_index<TargetType, OtherType, OtherTypes...> : std::integral_constant<int, 1 + type_index<TargetType, OtherTypes...>::value>
{};

//! Evaluates to true type if visit function is properly defined
template<typename VisitorT, typename VisitedT>
constexpr auto visitor_for_type_defined_test(int) -> decltype( std::declval<VisitorT>().visit(std::declval<VisitedT>()), std::true_type{} );

//! Evaluates to false type if visit function is not properly defined
template<typename VisitorT, typename VisitedT>
constexpr auto visitor_for_type_defined_test(long) -> decltype( std::false_type{} );

//! Evalutate if VisitorT has visit(VisitedT) defined. (const / reference must be set correctly for VisitedT)
template<typename VisitorT, typename VisitedT>
constexpr bool visitor_defined() {
    return decltype( visitor_for_type_defined_test<VisitorT, VisitedT>(0) )::value;
}


//! A recursive union to store / get types, not safe in itself,
//! relies on the variant class to handle it correctly
template<typename T, typename... OtherTypes>
union _recursive_union {

    //! Nothing constructor
    JUMP_INTEROPABLE
    _recursive_union()
    {}

    //! When the index reaches 0 (tag dispatch)
    #pragma nv_exec_check_disable
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    _recursive_union(
        std::integral_constant<int, 0>,
        ShouldBeT&& value
    )
        : value_( std::forward<ShouldBeT>(value) )
    {}

    //! Recursive assignment
    template<int current_index, typename NotT>
    JUMP_INTEROPABLE
    _recursive_union(
        std::integral_constant<int, current_index>,
        NotT&& value
    ):
        next_(
            std::integral_constant<int, current_index - 1>{},
            std::forward<NotT>(value)
        )
    {}

    //! When the index reaches 0 (tag dispatch)
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    _recursive_union(
        std::integral_constant<int,0>,
        const ShouldBeT& value
    )
        : value_( value )
    {}

    //! Recursive assignment
    template<int current_index, typename NotT>
    JUMP_INTEROPABLE
    _recursive_union(
        std::integral_constant<int, current_index>,
        const NotT& value
    ):
        next_(
            std::integral_constant<int, current_index - 1>{},
            value
        )
    {}

    //! Destruct - do nothing
    JUMP_INTEROPABLE
    ~_recursive_union() {
    }

    //! Copy assignment (termination case)
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    void assign(
        std::integral_constant<int, 0>,
        const ShouldBeT& value
    ) {
        new(&value_) ShouldBeT(value);
    }

    //! Copy assignment
    template<int current_index, typename NotT>
    JUMP_INTEROPABLE
    void assign(
        std::integral_constant<int, current_index>,
        const NotT& value
    ) {
        next_.assign(std::integral_constant<int, current_index - 1>{}, value);
    }

    //! move assignment (termination case)
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    void assign(
        std::integral_constant<int, 0>,
        ShouldBeT&& value
    ) {
        new(&value_) ShouldBeT(std::forward<ShouldBeT>(value));
    }

    //! move assignment
    template<int current_index, typename NotT>
    JUMP_INTEROPABLE
    void assign(
        std::integral_constant<int, current_index>,
        NotT&& value
    ) {
        next_.assign(std::integral_constant<int, current_index - 1>{}, std::forward<NotT>(value));
    }

    //! copy assignment
    #pragma nv_exec_check_disable
    template<typename recursive_union_t>
    JUMP_INTEROPABLE
    void assign(const int& index, const recursive_union_t& v) {
        if(index == 0) {
            new(&value_) T(v.value_);
        } else {
            next_.assign(index - 1, v.next_);
        }
    }

    //! Move assignment
    #pragma nv_exec_check_disable
    template<typename recursive_union_t>
    JUMP_INTEROPABLE
    void assign(const int& index, recursive_union_t&& v) {
        if(index == 0) {
            new(&value_) T(std::move(v.value_));
        } else {
            next_.assign(index - 1, std::move(v.next_));
        }
    }


    //! Get operator recurses to target_index == 0
    template<std::size_t target_index>
    JUMP_INTEROPABLE
    auto& get() {
        if constexpr(target_index == 0) {
            return value_;
        }
        if constexpr(target_index != 0) {
            return next_.template get<target_index - 1>();
        }
    }

    //! Get operator recurses to target_index == 0
    template<std::size_t target_index>
    JUMP_INTEROPABLE
    const auto& get() const {
        if constexpr(target_index == 0) {
            return value_;
        }
        if constexpr(target_index != 0) {
            return next_.template get<target_index - 1>();
        }
    }

    //! Destroy the value in the rescursive union
    #pragma nv_exec_check_disable
    JUMP_INTEROPABLE
    void destroy(const std::size_t& index) {
        if (index == 0) {
            value_.~T();
        } else {
            next_.destroy(index - 1);
        }
    }

    //! Visit this node
    #pragma nv_exec_check_disable
    template<typename visitor_t>
    JUMP_INTEROPABLE
    void visit(int index, visitor_t&& visitor) {
        if(index == 0) {
            if constexpr(_variant_helpers::visitor_defined<visitor_t, T&>()) {
                visitor.visit(value_);
            }
        } else {
            next_.template visit<visitor_t>(index - 1, std::move(visitor));
        }
    }

    //! Transfer to device
    void to_device(const std::size_t& index) {
        if(index == 0) {
            if constexpr(jump::class_interface<T>::to_device_defined()) {
                value_.to_device();
            }
        } else {
            next_.to_device(index - 1);
        }
    }

    //! Transfer from device
    void from_device(const std::size_t& index) {
        if(index == 0) {
            if constexpr(jump::class_interface<T>::from_device_defined()) {
                value_.from_device();
            }
        } else {
            next_.from_device(index);
        }
    }

    //! This value (if active)
    T value_; 
    //! Recursive union down to the other types
    _recursive_union<OtherTypes...> next_;
};


//! Terminating case for the final value
template<typename T>
union _recursive_union<T> {
    //! Nothing constructor
    JUMP_INTEROPABLE
    _recursive_union()
    {}

    //! Final value must be correct index
    #pragma nv_exec_check_disable
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    _recursive_union(
        std::integral_constant<int, 0>,
        ShouldBeT&& value
    ):
        value_(std::forward<ShouldBeT>(value))
    {}

    //! Destruct - do nothing
    JUMP_INTEROPABLE
    ~_recursive_union() {
    }

    //! Copy assignment (termination case)
    #pragma nv_exec_check_disable
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    void assign(
        std::integral_constant<int, 0>,
        const ShouldBeT& value
    ) {
        new(&value_) T(value);
    }

    //! move assignment (termination case)
    #pragma nv_exec_check_disable
    template<typename ShouldBeT>
    JUMP_INTEROPABLE
    void assign(
        std::integral_constant<int, 0>,
        ShouldBeT&& value
    ) {
        new(&value_) T(std::move(value_));
    }

    //! Copy assignment
    #pragma nv_exec_check_disable
    template<typename recursive_union_t>
    JUMP_INTEROPABLE
    void assign(const int& index, const recursive_union_t& v) {
        new(&value_) T(v.value_);
    }

    //! Move assignment
    #pragma nv_exec_check_disable
    template<typename recursive_union_t>
    JUMP_INTEROPABLE
    void assign(const int& index, recursive_union_t&& v) {
        new(&value_) T(std::move(v.value_));
    }

    //! Get the value at the final index
    template<std::size_t target_index>
    JUMP_INTEROPABLE
    auto& get() {
        return value_;
    }

    //! Get the value at the final index
    template<std::size_t target_index>
    JUMP_INTEROPABLE
    const auto& get() const {
        return value_;
    }

    //! Destory the value at index (must be this one if we reach here)
    #pragma nv_exec_check_disable
    JUMP_INTEROPABLE
    void destroy(const std::size_t& index) {
        value_.~T();
    }

    //! Visit this node (termination case)
    #pragma nv_exec_check_disable
    template<typename visitor_t>
    JUMP_INTEROPABLE
    void visit(int index, visitor_t&& visitor) {
        // must be the end
        if constexpr(_variant_helpers::visitor_defined<visitor_t, T&>()) {
            visitor.visit(value_);
        }
    }

    //! Transfer to device
    void to_device(const std::size_t& index) {
        if constexpr(jump::class_interface<T>::to_device_defined()) {
            value_.to_device();
        }
    }

    //! Transfer from device
    void from_device(const std::size_t& index) {
        if constexpr(jump::class_interface<T>::from_device_defined()) {
            value_.from_device();
        }
    }

    //! This value
    T value_;
};

} /* namespace _variant_helpers */

//! A quick variant implementation
template<typename... VariantTypes>
class variant {
public:
    //! Initialize an empty variant
    JUMP_INTEROPABLE
    variant():
        index_(-1)
    {}

    //! Move from value constructor
    template<typename T>
    JUMP_INTEROPABLE
    variant(T&& value):
        index_(_variant_helpers::type_index<std::decay_t<T>, VariantTypes...>::value),
        union_(std::integral_constant<int, _variant_helpers::type_index<std::decay_t<T>, VariantTypes...>::value>{}, std::forward<T>(value))
    {
        static_assert(_variant_helpers::type_index<std::decay_t<T>, VariantTypes...>::value >= 0, "Constructor type does not exist in variant");
    }

    //! Move from value constructor
    template<typename T>
    JUMP_INTEROPABLE
    variant(const T& value):
        index_(_variant_helpers::type_index<std::decay_t<T>, VariantTypes...>::value),
        union_(std::integral_constant<int, _variant_helpers::type_index<std::decay_t<T>, VariantTypes...>::value>{}, value)
    {
        static_assert(_variant_helpers::type_index<std::decay_t<T>, VariantTypes...>::value >= 0, "Constructor type does not exist in variant");
    }

    //! Copy destructor
    JUMP_INTEROPABLE
    variant(const variant<VariantTypes...>& other):
        index_(other.index_)
    {
        if(other.index_ >= 0)
            union_.assign(index_, other.union_);
    }
    
    //! Copy destructor
    JUMP_INTEROPABLE
    variant(variant<VariantTypes...>&& other):
        index_(other.index_)
    {
        if(other.index_ >= 0)
            union_.assign(index_, std::move(other.union_));
    }

    //! Copy assignment from other variant
    JUMP_INTEROPABLE
    variant<VariantTypes...>& operator=(const variant<VariantTypes...>& other) {
        if(index_ >= 0) {
            union_.destroy(index_);
        }

        index_ = other.index_;
        union_.assign(index_, other.union_);
        return *this;
    }

    //! Destruct the variant
    #pragma nv_exec_check_disable
    JUMP_INTEROPABLE
    ~variant() {
        //! Skip if we don't have anything
        if(index_ < 0) return;
        //! Otherwise use the union to delete it
        union_.destroy(index_);
    }

    //! Get the index of what we hold
    JUMP_INTEROPABLE
    auto index() const {
        return index_;
    }

    //! Gets contained element by index
    template<std::size_t target_index>
    JUMP_INTEROPABLE
    auto& get_index() {
        #if JUMP_ON_DEVICE
            assert( target_index == index_ && "Requested index must match what is stored");
        #else
            if(target_index != index_) {
                throw std::out_of_range("Requested index " + std::to_string(target_index) + " must match stored " + std::to_string(index_));
            }
        #endif
        return union_.template get<target_index>();
    }

    //! Gets contained element by index
    template<std::size_t target_index>
    JUMP_INTEROPABLE
    const auto& get_index() const {
        #if JUMP_ON_DEVICE
            assert( target_index == index_ && "Requested index must match what is stored");
        #else
            if(target_index != index_) {
                throw std::out_of_range("Requested index " + std::to_string(target_index) + " must match stored " + std::to_string(index_));
            }
        #endif
        return union_.template get<target_index>();
    }

    //! Get value by type specification
    template<typename TargetType>
    JUMP_INTEROPABLE
    auto& get() {
        static_assert(_variant_helpers::type_index<TargetType, VariantTypes...>::value >= 0, "Requested type does not exist in variant");
        return union_.template get<_variant_helpers::type_index<TargetType, VariantTypes...>::value>();
    }

    //! Get value by type specification
    template<typename TargetType>
    JUMP_INTEROPABLE
    const auto& get() const {
        static_assert(_variant_helpers::type_index<TargetType, VariantTypes...>::value >= 0, "Requested type does not exist in variant");
        return union_.template get<_variant_helpers::type_index<TargetType, VariantTypes...>::value>();
    }

    //! Visit value if defined
    template<typename visitor_t>
    JUMP_INTEROPABLE
    void visit(visitor_t&& visitor) {
        if(index_ < 0)
            return;
        union_.template visit<visitor_t>(index_, std::move(visitor));
    }

    // template<typename visitor_t>
    // JUMP_INTEROPABLE
    // void visit(visitor_t&& visitor) const {
        
    // }

    //! Transfer anything to device
    void to_device() {
        union_.to_device(index_);
    }

    //! Transfer anything to device
    void from_device() {
        union_.from_device(index_);
    }

private:
    //! Store the active type index
    int index_;
    //! Store the active type :)
    _variant_helpers::_recursive_union<VariantTypes...> union_;


}; /* class variant */

} /* namespace jump */

#endif /* JUMP_VARIANT_HPP_ */
