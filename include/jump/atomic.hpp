/**
 * @file atomic.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief Simple atomic wrapper to provide atomic and and subtract
 *  on GPU or CPU.
 * @date 2024-02-20
 */
#ifndef JUMP_ATOMIC_HPP_
#define JUMP_ATOMIC_HPP_

#include <jump/device_interface.hpp>

namespace jump {

/**
 * @brief a series of atomic operations defined on a templated type
 * @tparam T the type to creat an atomic of
 * @note not all types can be atomic, not all types that can be atomic have
 *  all the operators available for atomic operation.
 * @note As with many utilities here, implementation fleshing is driven by
 *  need and not everything that *could* be implemented *is*. Be careful
 *  friends :).
 * @note currently all CPU atomic operations use __ATOMIC_SEQ_CST
 * @note GPU atomic operations do NOT work on local memory,
 *  putting an atomic on the stack and using it will break things.
 *  Don't do that. I need to figure out a way to test for that, but
 *  nvcc will give a warning if you do at least.
 */
template<typename T>
class atomic {
public:
    //! Default constructor to default
    JUMP_INTEROPABLE
    atomic() = default;

    /**
     * @brief Construct a new atomic object, set value
     * @param v set the value to v
     */
    JUMP_INTEROPABLE
    atomic(const T& v):
        value(v)
    {}

    /**
     * @brief Construct an atomic from an atomic
     * @param a the atomic to pull from
     */
    JUMP_INTEROPABLE
    atomic(const atomic& a) {
        *this = a;
    }

    /////////////////// Function Implementations ////////////////////////

    /**
     * @brief atomic store
     * @param v value to set to
     * @note Uses atomicExch on GPU, Uses __atomic_store_n on CPU.
     * @note We do NOT implement any CAS clever things for GPU operation, only the NVIDIA
     *  implemented atomicExch types are valid on GPU.
     */
    JUMP_INTEROPABLE
    void store(const T& v) {
        #if JUMP_ON_DEVICE
            // This is fine for my immediate needs, but we should probably get more clever
            value = v;
            // the below doesn't work
            // if constexpr(sizeof(T) == sizeof(unsigned long long int)) {
            //     static_cast<T>(atomicExch(reinterpret_cast<unsigned long long int*>(&value), static_cast<unsigned long long int>(v)));
            // } else if constexpr(sizeof(T) == sizeof(unsigned int)) {
            //     static_cast<T>(atomicExch(reinterpret_cast<unsigned int*>(&value), static_cast<unsigned int>(v)));
            // } else if constexpr(sizeof(T) == sizeof(float)) {
            //     static_cast<T>(atomicExch(reinterpret_cast<float*>(&value), static_cast<float>(v)));
            // } else {
            //     atomicExch(&value, v);
            // }
        #else
            // TODO: verify / fix this!
            if constexpr(std::is_same<float, T>::value) {
                __atomic_store_n(reinterpret_cast<int32_t*>(const_cast<float*>(&value)), *reinterpret_cast<int32_t*>(const_cast<float*>(&v)), __ATOMIC_SEQ_CST);
            } else {
                __atomic_store_n(&value, v, __ATOMIC_SEQ_CST);
            }
        #endif
    }

    /**
     * @brief loads value
     * @return the held value
     * @note Uses atomicCAS on GPU, __atomic_load on CPU
     */
    JUMP_INTEROPABLE
    T load() const {
        T r;
        #if JUMP_ON_DEVICE
            // This is bad and should probably be done differently lol
            // if constexpr(sizeof(T) == sizeof(unsigned int)) {
            //     r = static_cast<T>(atomicAdd(reinterpret_cast<unsigned int*>(const_cast<T*>(&value)), 0));
            // } else if constexpr(sizeof(T) == sizeof(unsigned long long int)) {
            //     r = static_cast<T>(atomicAdd(reinterpret_cast<unsigned long long int*>(const_cast<T*>(&value)), 0));
            // } else if constexpr(sizeof(T) == sizeof(int)) {
            //     r = static_cast<T>(atomicAdd(reinterpret_cast<int*>(const_cast<T*>(&value)), 0));
            // } else {
            //     r = atomicAdd(&value, 0);
            // }
            r = atomicAdd(const_cast<T*>(&value), static_cast<T>(0));
            // if constexpr(sizeof(T) == sizeof(unsigned int)) {
            //     unsigned int cmp = 0;
            //     r = static_cast<T>(atomicCAS(reinterpret_cast<unsigned int*>(const_cast<T*>(&value)), cmp, cmp));
            // } else if constexpr(sizeof(T) == sizeof(unsigned long long int)) {
            //     unsigned long long int cmp = 0;
            //     r = static_cast<T>(atomicCAS(reinterpret_cast<unsigned long long int*>(const_cast<T*>(&value)), cmp, cmp));
            // } else if constexpr(sizeof(T) == sizeof(int)) {
            //     int cmp = 0;
            //     r = static_cast<T>(atomicCAS(reinterpret_cast<int*>(const_cast<T*>(&value)), cmp, cmp));
            // } else {
            //     T cmp = 0;
            //     r = atomicCAS(&value, cmp, cmp);
            // }
        #else
            __atomic_load(&value, &r, __ATOMIC_SEQ_CST);
        #endif
        return r;
    }

    /**
     * @brief adds a value to this atomic and returns the value before the add
     * @param v the value to add
     * @return value before the add
     */
    JUMP_INTEROPABLE
    T fetch_add(const T& v) {
        #if JUMP_ON_DEVICE
            auto before = value;
            auto r = atomicAdd(&value, v);
            return r;
        #else
            if constexpr(std::is_same<float, T>::value) {
                auto p = load();
                store(load() + v);
                return p;
            } else {
                // TODO: fix this!!! https://stackoverflow.com/questions/45055402/atomic-double-floating-point-or-sse-avx-vector-load-store-on-x86-64
                return __atomic_fetch_add(&value, v, __ATOMIC_SEQ_CST);
            }
        #endif
    }

    /**
     * @brief subtracts a value from this atomic and returns the value before the add
     * @param v the value to subtract
     * @return value before the subtract
     */
    JUMP_INTEROPABLE
    T fetch_subtract(const T& v) {
        #if JUMP_ON_DEVICE
            return atomicSub(&value, v);
        #else
            return __atomic_fetch_sub(&value, v, __ATOMIC_SEQ_CST);
        #endif
    }

    /**
     * @brief and the value against v
     * @param v the value to and against
     * @return value before the and
     */
    JUMP_INTEROPABLE
    T fetch_and(const T& v) {
        #if JUMP_ON_DEVICE
            return atomicAnd(&value, v);
        #else
            return __atomic_fetch_and(&value, v, __ATOMIC_SEQ_CST);
        #endif
    }

    /**
     * @brief or the value against v
     * @param v the value to or against
     * @return value before the or
     */
    JUMP_INTEROPABLE
    T fetch_or(const T& v) {
        #if JUMP_ON_DEVICE
            return atomicOr(&value, v);
        #else
            return __atomic_fetch_or(&value, v, __ATOMIC_SEQ_CST);
        #endif
    }

    /**
     * @brief or the value against v
     * @param v the value to or against
     * @return value before the or
     */
    JUMP_INTEROPABLE
    T fetch_xor(const T& v) {
        #if JUMP_ON_DEVICE
            return atomicXor(&value, v);
        #else
            return __atomic_fetch_xor(&value, v, __ATOMIC_SEQ_CST);
        #endif
    }

    /**
     * @brief compute min against a value
     * @param v the value to compute min against
     * @return the value before the min operator
     * @note on GPU uses atomicMin, on CPU uses atomic compare exchange with a loop.
     */
    JUMP_INTEROPABLE
    T fetch_min(const T& v) {
        #if JUMP_ON_DEVICE
            if constexpr(std::is_same<T, float>::value) {
                // if(v < value) {
                //     value = v;
                // }
                // return value;
                int ret = __float_as_int(value);
                while(v < __int_as_float(ret))
                {
                    int old = ret;
                    if((ret = atomicCAS(reinterpret_cast<int*>(&value), old, __float_as_int(v))) == old)
                        break;
                }
                return __int_as_float(ret);
            } else {
                return atomicMin(&value, v);
            }
        #else
            T prev_value = load();
            while(
                prev_value > v &&
                !__atomic_compare_exchange(&value, &prev_value, &v, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
            ) {}
            return prev_value;
        #endif
    }

    /**
     * @brief compute min against a value
     * @param v the value to compute min against
     * @return the value before the min operator
     * @note on GPU uses atomicMin, on CPU uses atomic compare exchange with a loop.
     */
    JUMP_INTEROPABLE
    T fetch_max(const T& v) {
        #if JUMP_ON_DEVICE
            return atomicMax(&value, v);
        #else
            T prev_value = load();
            while(
                prev_value < v &&
                !__atomic_compare_exchange(&value, &prev_value, &v, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
            ) {}
            return prev_value;
        #endif
    }

    /////////////////// Operator interface ////////////////////////

    /**
     * @brief atomic assignment operator.
     * @param v the value to set to (gets value using load())
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    atomic& operator=(const atomic& v) {
        store(v.load());
        return *this;
    }

    /**
     * @brief atomic assignment operator.
     * @param v the value to set to
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    atomic& operator=(const T& v) {
        store(v);
        return *this;
    }

    /**
     * @brief atomic addition
     * @param v value to add
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    atomic& operator+=(const T& v) {
        fetch_add(v);
        return *this;
    }

    /**
     * @brief atomic subtraction
     * @param v value to subtract
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    atomic& operator-=(const T& v) {
        fetch_subtract(v);
        return *this;
    }

    /**
     * @brief atomic increment operator (prefix)
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    atomic& operator++() {
        fetch_add(1);
        return *this;
    }

    /**
     * @brief atomic increment operator (postfix)
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    T operator++(int) {
        return fetch_add(1);
    }
    /**
     * @brief atomic decrement operator (prefix)
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    atomic& operator--() {
        fetch_subtract(1);
        return *this;
    }

    /**
     * @brief atomic decrement operator (postfix)
     * @return this atomic by reference
     */
    JUMP_INTEROPABLE
    T operator--(int) {
        return fetch_subtract(1);
    }

    /**
     * @brief get the underlying value
     * @return T the contained value.
     */
    JUMP_INTEROPABLE
    operator T() const {
        return load();
    }

private:
    //! The value that is being held (all operations will operate on this location in memory)
    T value;

}; /* class atomic */


} /* namespace jump */

#endif /* JUMP_ATOMIC_HPP_ */
