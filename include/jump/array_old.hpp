/**
 * @file array.hpp
 * @author Joshua Spisak
 * @brief implementation of jump::array (a 1d interopable array data-structure) and some helpers.
 * @date 2022-05-07
 */
#ifndef JUMP_ARRAY_HPP_
#define JUMP_ARRAY_HPP_

#include "jump/cuda_interface.hpp"
#include "jump/shared_ptr.hpp"

#include <vector>

//! top-level utils namespace
namespace jump {

//! An internal namespace for the array class to use, this is very similar
//! to shared_ptr_helpers... abstract to another header of shared_data_helpers.hpp?
namespace array_helpers {

    //! Used to constexpr check if toGPU(T*) exists in a class
    template <typename>
    constexpr std::false_type toGPUDefinedT (long);

    //! Used to constexpr check if toGPU(T*) exists in a class
    template <typename T>
    constexpr auto toGPUDefinedT (int)
    -> decltype( std::declval<T>().toGPU(nullptr), std::true_type{} );

    //! Used to constexpr check if toGPU() exists in a class
    template <typename T>
    using toGPUDefined = decltype( toGPUDefinedT<T>(0) );

    //! Used to constexpr check if fromGPU(T*) exists in a class
    template <typename>
    constexpr std::false_type fromGPUDefinedT (long);

    //! Used to constexpr check if fromGPU(T*) exists in a class
    template <typename T>
    constexpr auto fromGPUDefinedT (int)
    -> decltype( std::declval<T>().fromGPU(nullptr), std::true_type{} );

    //! Used to constexpr check if fromGPU(T*) exists in a class
    template <typename T>
    using fromGPUDefined = decltype( fromGPUDefinedT<T>(0) );


    //! Used to constexpr check if toGPU(void) exists in a class
    template <typename>
    constexpr std::false_type toGPUVoidDefinedT (long);

    //! Used to constexpr check if toGPU(void) exists in a class
    template <typename T>
    constexpr auto toGPUVoidDefinedT (int)
    -> decltype( std::declval<T>().toGPU(), std::true_type{} );

    //! Used to constexpr check if toGPU(void) exists in a class
    template <typename T>
    using toGPUVoidDefined = decltype( toGPUVoidDefinedT<T>(0) );

    //! Used to constexpr check if fromGPU(void) exists in a class
    template <typename>
    constexpr std::false_type fromGPUVoidDefinedT (long);

    //! Used to constexpr check if fromGPU(void) exists in a class
    template <typename T>
    constexpr auto fromGPUVoidDefinedT (int)
    -> decltype( std::declval<T>().fromGPU(), std::true_type{} );

    //! Used to constexpr check if fromGPU(void) exists in a class
    template <typename T>
    using fromGPUVoidDefined = decltype( fromGPUVoidDefinedT<T>(0) );

} /* namespace array_helpers */

/**
 * @brief creates an array that is shared across host and device or in unified memory
 * @tparam T underlaying type this is an array of
 */
template<typename T>
class array {
public:
    //! Utility type definition
    using sptr = shared_ptr<array>;

    //! Make a shared pointer on the host
    template<typename... Args>
    static sptr ptr(Args&&... args) {
        return make_shared_on<array>(MemType::HOST, args...);
    }

    //! Help with template deduction :)
    template<typename... Args>
    static sptr ptr(std::initializer_list<T>&& data, Args&&... args) {
        return make_shared_on<array>(MemType::HOST, data, args...);
    }

    //! Make a shared pointer in unified memory
    template<typename... Args>
    static sptr unified_ptr(Args&&... args) {
        return make_shared_on<array>(MemType::UNIFIED, args..., MemType::UNIFIED);
    }

    //! Help with template deduction :)
    template<typename... Args>
    static sptr unified_ptr(std::initializer_list<T>&& data, Args&&... args) {
        return make_shared_on<array>(MemType::UNIFIED, data, args..., MemType::UNIFIED);
    }

    //! Create a unified side pointer to a device side array
    template<typename... Args>
    static sptr device_ptr(Args&&... args) {
        return make_shared_on<array>(MemType::UNIFIED, args..., MemType::DEVICE);
    }

    //! Help with template deduction :)
    template<typename... Args>
    static sptr device_ptr(std::initializer_list<T>&& data, Args&&... args) {
        return make_shared_on<array>(MemType::UNIFIED, data, args..., MemType::DEVICE);
    }

    //! Create a host side pointer to a device side array
    template<typename... Args>
    static sptr host_to_device_ptr(Args&&... args) {
        return make_shared_on<array>(MemType::HOST, args..., MemType::DEVICE);
    }

    //! Help with template deduction :)
    template<typename... Args>
    static sptr host_to_device_ptr(std::initializer_list<T>&& data, Args&&... args) {
        return make_shared_on<array>(MemType::HOST, data, args..., MemType::DEVICE);
    }

    /**
     * @brief Construct an empty host array
     */
    array():
        primary_location_(MemType::HOST),
        data_(nullptr),
        device_data_(nullptr),
        size_(0)
    {}

    /**
     * @brief Construct a new shared array object from an initializer list
     * @param data members to initialize data from
     * @param location what memory to create this array in
     */
    array(
        const std::initializer_list<T>& data,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_(data.size())
    {
        allocate();

        if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                std::size_t i = 0;
                for(auto& obj : data) {
                    cudaMemcpy(device_data_ + i, &obj, sizeof(T), cudaMemcpyHostToDevice);
                    ++i;
                }
            #endif
        } else {
            std::size_t i = 0;
            for(auto& obj : data) {
                new(&data_[i]) T(obj);
                ++i;
            }
        }
    }

    /**
     * @brief Construct a new shared array object from a vector
     * @param data members to initialize data from
     * @param location what memory to create this array in
     */
    array(
        const std::vector<T>& data,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_(data.size())
    {
        allocate();

        if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                std::size_t i = 0;
                for(auto& obj : data) {
                    cudaMemcpy(device_data_ + i, &obj, sizeof(T), cudaMemcpyHostToDevice);
                    ++i;
                }
            #endif
        } else {
            std::size_t i = 0;
            for(auto& obj : data) {
                new(&data_[i]) T(obj);
                ++i;
            }
        }
    }
    
    /**
     * @brief Construct a new shared array object with a fixed size, no construction of objects
     * @param size of the array
     * @param location what memory to create this array in
     */
    array(
        std::size_t size,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_(size)
    {
        allocate();

        // No construction / initialization to do
    }
    
    /**
     * @brief Construct a new shared array object of a size with a default value
     * @param size number of elements in array
     * @param default_value the default value to initialize all elements from
     * @param location where to construct the array
     */
    array(
        std::size_t size,
        const T& default_value,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_(size)
    {
        allocate();

        // checking for CUDA happens in allocate(), so we just
        // need to make sure this compiles if not CUDA_AVAILABLE
        if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                for(std::size_t i = 0; i < size_; ++i) {
                    cudaMemcpy(device_data_ + i, &default_value, sizeof(T), cudaMemcpyHostToDevice);
                }
            #endif
        } else {
            for(std::size_t i = 0; i < size_; ++i) {
                new(&data_[i]) T(default_value);
            }
        }
    }

    /**
     * @brief Construct a new shared array object via a copy constructor
     * @param obj object to copy from
     */
    array(const array& obj):
        primary_location_(obj.primary_location_),
        data_(nullptr),
        device_data_(nullptr),
        size_(obj.size_)
    {
        allocate();

        if(primary_location_ != MemType::DEVICE) {
            for(std::size_t i = 0; i < size_; ++i) {
                new(&data_[i]) T(obj.data_[i]);
            }
        } else {
            // Mixed messaging over whether this is the right move
            // https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
            #if CUDA_AVAILABLE
                cudaMemcpy(device_data_, obj.device_data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice);
            #endif
        }

        if(primary_location_ == MemType::HOST && obj.device_data_ != nullptr) {
            toGPU();
        }
        if(primary_location_ == MemType::DEVICE && obj.data_ != nullptr) {
            fromGPU();
        }
    }


    /**
     * @brief Construct a new shared array object via a copy constructor
     * @param obj object to copy from
     */
    array(array&& obj):
        primary_location_(obj.primary_location_),
        data_(obj.data_),
        device_data_(obj.device_data_),
        size_(obj.size_)
    {
        obj.primary_location_ = MemType::HOST;
        obj.data_ = nullptr;
        obj.device_data_ = nullptr;
        obj.size_ = 0;
    }
    
    array& operator=(const array& obj)
    {
        deallocate();
        primary_location_ = obj.primary_location_;
        data_ = nullptr;
        device_data_ = nullptr;
        size_ = obj.size_;
        allocate();

        if(primary_location_ != MemType::DEVICE) {
            for(std::size_t i = 0; i < size_; ++i) {
                new(&data_[i]) T(obj.data_[i]);
            }
        } else {
            // Mixed messaging over whether this is the right move
            // https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
            #if CUDA_AVAILABLE
                cudaMemcpy(device_data_, obj.device_data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice);
            #endif
        }

        if(primary_location_ == MemType::HOST && obj.device_data_ != nullptr) {
            toGPU();
        }
        if(primary_location_ == MemType::DEVICE && obj.data_ != nullptr) {
            fromGPU();
        }
        return *this;
    }

    array& operator=(array&& obj) {
        primary_location_ = obj.primary_location_;
        data_ = obj.data_;
        device_data_ = obj.device_data_;
        size_ = obj.size_;

        obj.primary_location_ = MemType::HOST;
        obj.data_ = nullptr;
        obj.device_data_ = nullptr;
        obj.size_ = 0;
        return *this;
    }

    /**
     * @brief Destroy the shared array object, deallocate any memory
     */
    ~array() {
        deallocate();
    }

    /**
     * @brief get the size of the array
     * @return std::size_t number of elements in array
     */
    GPU_COMPATIBLE
    std::size_t size() const {
        return size_;
    }

    /**
     * @brief directly access the data in the container
     * @return const T* pointer to data
     */
    GPU_COMPATIBLE
    T* data() const {
        #if ON_GPU
            return device_data_;
        #else
            return data_;
        #endif
    }

    /**
     * @brief directly access device data in the container
     * @return const T* pointer to device data
     */
    GPU_COMPATIBLE
    T* data_device() const {
        return device_data_;
    }

    GPU_COMPATIBLE
    T& at(std::size_t index) const {
        #if ON_GPU
            return device_data_[index];
        #else
            if(data_ == nullptr) throw std::runtime_error("data_ == nullptr, unable to index on CPU");
            if(index >= size_) throw std::out_of_range("index is out of range in array");
            return data_[index];
        #endif
    }

    GPU_COMPATIBLE
    T& operator[](std::size_t index) const {
        return at(index);
    }

    std::vector<T> vector() {
        std::vector<T> result(size_);
        if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                cudaMemcpy(&result[0], device_data_, sizeof(T) * size_, cudaMemcpyDeviceToHost);
            #else
                throw std::runtime_error("CUDA is not available, unable to create vector from array.");
            #endif
        } else {
            for(std::size_t i = 0; i < size_; ++i) {
                result[i] = data_[i];
            }
        }
        return result;
    }


    /**
     * @brief transfer data from HOST to DEVICE (no effect if UNIFIED)
     */
    void fromGPU() {
        #if CUDA_AVAILABLE
            if (primary_location_ == MemType::DEVICE) {
                if(data_ == nullptr) {
                    data_ = static_cast<T*>(malloc(sizeof(T) * size_));
                }

                cudaMemcpy(data_, device_data_, sizeof(T) * size_, cudaMemcpyDeviceToHost);
            } else if(primary_location_ == MemType::HOST) {
                if(device_data_ == nullptr) {
                    throw std::runtime_error("DEVICE memory is not allocated, unable to copy from GPU");
                }
                if constexpr(array_helpers::fromGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_; ++i) {
                        bool copy = true;
                        copy = data_[i]->fromGPU(device_data_ + i);
                        if(copy)
                            cudaMemcpy(&data_[i], device_data_ + i, sizeof(T), cudaMemcpyDeviceToHost);
                    }
                } else {
                    if constexpr(array_helpers::fromGPUVoidDefined<T>{}) {
                        for(std::size_t i = 0; i < size_; ++i) {
                            data_[i]->fromGPU();
                        }
                    }
                    cudaMemcpy(data_, device_data_, sizeof(T) * size_, cudaMemcpyDeviceToHost);
                }
            } else if(primary_location_ == MemType::UNIFIED) {
                if constexpr(array_helpers::fromGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_; ++i)
                        data_[i].fromGPU(nullptr);
                } else if constexpr(array_helpers::fromGPUVoidDefined<T>{}) {
                    for(std::size_t i = 0; i < size_; ++i)
                        data_[i].fromGPU();
                }
            }
        #else
            throw std::runtime_error("Unable to transfer to GPU when CUDA is not available.");
        #endif
    }

    /**
     * @brief transfer data from DEVICE to HOST (no effect if UNIFIED)
     */
    void toGPU() {
        #if CUDA_AVAILABLE
            if (primary_location_ == MemType::DEVICE) {
                if(data_ == nullptr) {
                    throw std::runtime_error("Host memory is not allocated, unable to copy to GPU");
                }

                cudaMemcpy(device_data_, data_, sizeof(T) * size_, cudaMemcpyHostToDevice);
            } else if(primary_location_ == MemType::HOST) {
                if(device_data_ == nullptr) {
                    cudaMalloc(&device_data_, sizeof(T) * size_);
                }
                if constexpr(array_helpers::toGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_; ++i) {
                        bool copy = true;
                        copy = data_[i]->toGPU(device_data_ + i);
                        if(copy)
                            cudaMemcpy(device_data_ + i, &data_[i], sizeof(T), cudaMemcpyHostToDevice);
                    }
                } else {
                    if constexpr(array_helpers::toGPUVoidDefined<T>{}) {
                        for(std::size_t i = 0; i < size_; ++i) {
                            data_[i]->toGPU();
                        }
                    }
                    cudaMemcpy(device_data_, data_, sizeof(T) * size_, cudaMemcpyHostToDevice);
                }
            } else if(primary_location_ == MemType::UNIFIED) {
                if constexpr(array_helpers::toGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_; ++i)
                        data_[i].toGPU(nullptr);
                } else if constexpr(array_helpers::toGPUVoidDefined<T>{}) {
                    for(std::size_t i = 0; i < size_; ++i)
                        data_[i].toGPU();
                }
            }
        #else
            throw std::runtime_error("Unable to transfer to GPU when CUDA is not available.");
        #endif
    }


private:
    void deallocate() {
        // We only run the destructor on objects in
        // HOST or UNIFIED memory
        if(primary_location_ != MemType::DEVICE) {
            for(std::size_t i = 0; i < size_; ++i) {
                data_[i].~T();
            }
        }

        if(primary_location_ == MemType::UNIFIED) {
            #if CUDA_AVAILABLE
                if(device_data_ != nullptr) {
                    cudaFree(device_data_);
                    device_data_ = nullptr;
                    data_ = nullptr;
                }
            #else
                std::fprintf(stderr, "ERROR: de-allocating unified array, but CUDA is not available, ensure full project is compiled with CUDA.\n");
            #endif
        } else {
            if(data_ != nullptr) {
                free(data_);
                data_ = nullptr;
            }
            if(device_data_ != nullptr) {
                #if CUDA_AVAILABLE
                    cudaFree(device_data_);
                    device_data_ = nullptr;
                #else
                    std::fprintf(stderr, "ERROR: device_data_ != nullptr, but CUDA is not available, ensure full project is compiled with CUDA.\n");
                #endif
            }
        }
    }

    void allocate() {
        if(size_ == 0) return;

        if(primary_location_ == MemType::HOST) {
            data_ = static_cast<T*>(malloc(sizeof(T) * size_));
        } else if(primary_location_ == MemType::UNIFIED) {
            #if CUDA_AVAILABLE
                cudaMallocManaged(&data_, sizeof(T) * size_);
                device_data_ = data_;
            #else
                throw std::runtime_error("CUDA Not available, unable to allocated unified memory");
            #endif
        } else if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                cudaMalloc(&device_data_, sizeof(T) * size_);
            #else
                throw std::runtime_error("CUDA Not available, unable to allocated unified memory");
            #endif
        } else {
            throw std::runtime_error("Unknown primary location for data. This should not be happening! You should panic!");
        }
    }

    //! The primary location of the data, this effects how some
    //! operations interfacing with vectors work
    MemType primary_location_;
    //! Array allocated on host 
    T* data_;
    //! Array allocated on device
    T* device_data_;
    //! Size of the array
    std::size_t size_;

}; /* class array */

} /* namespace jump */


#endif /* JUMP_ARRAY_HPP_ */