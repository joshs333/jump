/**
 * @file device_interface.hpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief utilities to see if devices are available (at build / compile time)
 *  and macros / enums to simplify device interfacing / interopability
 * @date 2022-07-25
 */
#ifndef JUMP_DEVICE_INTERFACE_HPP_
#define JUMP_DEVICE_INTERFACE_HPP_

// JUMP_ENABLE_CUDA means we enable cuda at compile-time, but it is
// possible for CUDA to be disabled at runtime if cudart is not available
// or there are no GPU's detected
#ifdef JUMP_ENABLE_CUDA
    // #include <cuda.h>
    // #include <device_functions.h>
    #include <cuda_runtime_api.h>
    #include <dlfcn.h>
#endif

#include <type_traits>


// If cuda is enabled and we are compiling with a device compatible 
#if defined(JUMP_ENABLE_CUDA) && defined(__CUDACC__)
    #define JUMP_DEVICE_ONLY __device__
    #define JUMP_INTEROPABLE __host__ __device__
#else
    #define JUMP_DEVICE_ONLY
    #define JUMP_INTEROPABLE
#endif

// A slightly more friendly macro to check if code is being built
// for device or host (check within a function)
// NOTE: on_device() below is a cleaner way to if constexpr evaluate if running on device
#ifdef __CUDA_ARCH__
    #define JUMP_ON_DEVICE true
#else
    #define JUMP_ON_DEVICE false
#endif

//! Just Multi-Processing namespace
namespace jump {

//! Internal variable for if the device count is evaluated
bool _cuda_eval = false;
//! Internal variable for if cuda is available (valid after _cuda_eval = true)
bool _cuda_available = false;
//! Internal variable for number of devices available (valid after _cuda_eval = true)
int _device_count = 0;

/**
 * @brief a constexpr evaluation of if CUDA is enabled
 * @note even if cuda is enabled, there might not be devices to run on,
 *  check that with devices_available()
 * @return true or false if cuda is enabled based on definitions
 */
constexpr bool cuda_enabled() {
    #ifdef JUMP_ENABLE_CUDA
        return true;
    #else
        return false;
    #endif
} /* cuda_enabled() */

/**
 * @brief function to determine if there are devices available at runtime
 * @return true or false if the number of devices are non-zero
 */
inline bool devices_available() {
    if constexpr(cuda_enabled()) {
        auto r = cudaGetDeviceCount(&_device_count);
        if(r != 0)
            return _cuda_available;
        if(_device_count > 0)
            _cuda_available = true;
        return _cuda_available;
    #else
        return false;
    #endif

    // We don't need to do this because of static linkage - but if we wanted
    // we could do runtime library loading / checking
    /*
    if (_cuda_eval)
        return _cuda_available;

    // function must leave _cuda_available and _device_count in an accurate state
    _cuda_eval = true;

    // if we can't find the rt library assume cuda is not available
    void* handle = dlopen("libcudart.so", RTLD_LAZY);
    if (!handle)
        return _cuda_available;
    
    typedef int (*hello_t)( int* count );

    // if we can't find the function that's extremely weird... but assume cuda is not available
    dlerror();
    hello_t hello = (hello_t) dlsym(handle, "cudaGetDeviceCount");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        dlclose(handle);
        return _cuda_available;
    }

    auto cuda_error = hello(&_device_count);
    dlclose(handle);

    if(static_cast<int>(cuda_error) != 0)
        return _cuda_available;

    if(_device_count > 0) 
        _cuda_available = true;
    return _cuda_available;
    */

} /* cuda_available() */

/**
 * @brief gets the number of devices (GPU) detected by cuda
 * @return int the number of devices available
 */
inline int device_count() {
    #ifdef JUMP_ENABLE_CUDA
        if(!_cuda_eval)
            cuda_available();
        return _device_count;
    #else
        return 0;
    #endif
} /* cuda_device_count() */

/**
 * @brief allow a constexpr if instead of a macro call to determine
 *  if code is being executed on device
 * @return true if running on device, false if running on host
 **/
constexpr bool on_device() {
    #if JUMP_ON_DEVICE
        return true;
    #else
        return false;
    #endif
} /* on_device() */

/**
 * @brief adds a thread synchronization point that only
 *  works if code is executing on device
 * @note it might be interesting to explore what these 
 *  synchronization mechanisms might mean on host and
 *  implement an interopable synchronization mechanism
 **/
JUMP_INTEROPABLE
inline void device_thread_sync() {
    // This doesn't work! Which is weird, it seems
    // to work fine for switching between __host__ and
    // __device__ functions, I guess because this is a
    // primitive it's different?
    // if constexpr(on_device()) {
    //     __syncthreads();
    // }
    // Whatever, for this we will just do a macro if
    #if JUMP_ON_DEVICE
        __syncthreads();
    #endif
} /* device_thread_sync() */

//! A few functions to help perform constexpr evaluation of types
//! to determine compatibility / interfacing
namespace _device_interface_helpers {
    // This overload of to_device_defined_overload is selected if T.to_device() exists
    template <typename T>
    constexpr auto to_device_defined_overload(int) -> decltype( std::declval<T>().to_device(), std::true_type{} );

    // This overload is selected if the above overload fails (T.to_device() expression is invalid)
    template <typename>
    constexpr auto to_device_defined_overload(long) -> std::false_type;

    // This evaluates if to_device() is defined using the overloads above
    template <typename T>
    using to_device_defined_test = decltype( to_device_defined_overload<T>(0) );

    // This overload of from_device_defined_overload is selected if T.from_device() exists
    template <typename T>
    constexpr auto from_device_defined_overload(int) -> decltype( std::declval<T>().from_device(), std::true_type{} );

    // This overload is selected if the above overload fails (T.from_device() expression is invalid)
    template <typename>
    constexpr auto from_device_defined_overload(long) -> std::false_type;

    // This evaluates if from_device() is defined using the overloads above
    template <typename T>
    using from_device_defined_test = decltype( from_device_defined_overload<T>(0) );

    // This overload of host_compatible_defined_overload is selected if T::host_compatible exists
    template <typename T>
    constexpr auto host_compatible_defined_overload(int) -> decltype( T::host_compatible, std::true_type{} );

    // This overload is selected if the above overload fails (T::host_compatible expression is invalid)
    template <typename>
    constexpr auto host_compatible_defined_overload(long) -> std::false_type;

    // This evaluates if T::host_compatible is defined using the overloads above
    template <typename T>
    using host_compatible_defined_test = decltype( host_compatible_defined_overload<T>(0) );

    // This overload of from_device_defined_overload is selected if T::device_compatible exists
    template <typename T>
    constexpr auto device_compatible_defined_overload(int) -> decltype( T::device_compatible, std::true_type{} );

    // This overload is selected if the above overload fails (T.from_device() expression is invalid)
    template <typename>
    constexpr auto device_compatible_defined_overload(long) -> std::false_type;

    // This evaluates if T::device_compatible is defined using the overloads above
    template <typename T>
    using device_compatible_defined_test = decltype( device_compatible_defined_overload<T>(0) );
} /* namespace _device_interface_helpers */


/**
 * @brief used to evaluate the interfacing / compatibility of a type for
 *  usage as a kernel
 * @tparam T the type to evaluate
 */
template<typename T>
struct kernel_interface {
    /**
     * @brief determine whether the to_device() function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool to_device_defined() {
        return _device_interface_helpers::to_device_defined_test<T>::value;
    }

    /**
     * @brief determine whether the to_device() function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool from_device_defined() {
        return _device_interface_helpers::from_device_defined_test<T>::value;
    }
    
    /**
     * @brief determines if a class is host compatible (can be executed on the host)
     * @return the value of T::host_compatible if it exists, otherwise default to true
     */
    static constexpr bool host_compatible() {
        if constexpr(_device_interface_helpers::host_compatible_defined_test<T>::value) {
            return T::host_compatible;
        }
        return true;
    }
    
    /**
     * @brief determines if a class is device compatible (can be executed on the device)
     * @return the value of T::device_compatible if it exists, otherwise default to true
     */
    static constexpr bool device_compatible() {
        if constexpr(_device_interface_helpers::device_compatible_defined_test<T>::value) {
            return T::device_compatible;
        }
        return true;
    }
}; /* struct kernel_interface */


/**
 * @brief used to evaluate the interfacing / compatibility of a type
 *  for usage in interopable data structures
 * @tparam T the type to evaluate
 */
template<typename T>
struct class_interface {
    /**
     * @brief determine whether the to_device() function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool to_device_defined() {
        return _device_interface_helpers::to_device_defined_test<T>::value;
    }

    /**
     * @brief determine whether the to_device() function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool from_device_defined() {
        return _device_interface_helpers::from_device_defined_test<T>::value;
    }
}; /* struct class_interface */

} /* namespace jump */

#endif /* JUMP_DEVICE_INTERFACE_HPP_ */
