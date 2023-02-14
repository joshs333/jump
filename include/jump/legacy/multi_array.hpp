/**
 * @file multi_array.hpp
 * @author Joshua Spisak
 * @brief implementation of jump::multi_array (a 2d interopable array data-structure) and some helpers.
 * @date 2022-05-07
 */
#ifndef JUMP_MULTI_ARRAY_HPP_
#define JUMP_MULTI_ARRAY_HPP_

#include "jump/cuda_interface.hpp"
#include "jump/shared_ptr.hpp"

//! top-level utils namespace
namespace jump {

//! An internal namespace for the multi_array class to use, this is very similar
//! to shared_ptr_helpers... abstract to another header of shared_data_helpers.hpp?
namespace multi_array_helpers {

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

} /* namespace multi_array_helpers */

/**
 * @brief creates an multi_array that is shared across host and device or in unified memory
 * @tparam T underlaying type this is an multi_array of
 */
template<typename T>
class multi_array {
public:
    //! Utility type definition
    using sptr = shared_ptr<multi_array>;

    //! Make a shared pointer on the host
    template<typename... Args>
    static sptr ptr(Args&&... args) {
        return make_shared_on<multi_array>(MemType::HOST, args...);
    }

    //! Help with template deduction :)
    template<typename... Args>
    static sptr ptr(std::initializer_list<std::initializer_list<T>>&& data, Args&&... args) {
        return make_shared_on<multi_array>(MemType::HOST, data, args...);
    }

    //! Make a shared pointer in unified memory
    template<typename... Args>
    static sptr unified_ptr(Args&&... args) {
        return make_shared_on<multi_array>(MemType::UNIFIED, args..., MemType::UNIFIED);
    }

    //! Help template deduction :)
    template<typename... Args>
    static sptr unified_ptr(std::initializer_list<std::initializer_list<T>>&& data, Args&&... args) {
        return make_shared_on<multi_array>(MemType::UNIFIED, data, args..., MemType::UNIFIED);
    }

    //! Create a unified side pointer to a device side array
    template<typename... Args>
    static sptr device_ptr(Args&&... args) {
        return make_shared_on<multi_array>(MemType::UNIFIED, args..., MemType::DEVICE);
    }

    //! Help template deduction :)
    template<typename... Args>
    static sptr device_ptr(std::initializer_list<std::initializer_list<T>>&& data, Args&&... args) {
        return make_shared_on<multi_array>(MemType::UNIFIED, data, args..., MemType::DEVICE);
    }

    //! Create a host side pointer to a device side array
    template<typename... Args>
    static sptr host_to_device_ptr(Args&&... args) {
        return make_shared_on<multi_array>(MemType::HOST, args..., MemType::DEVICE);
    }

    //! Help template deduction :)
    template<typename... Args>
    static sptr host_to_device_ptr(std::initializer_list<std::initializer_list<T>>&& data, Args&&... args) {
        return make_shared_on<multi_array>(MemType::HOST, data, args..., MemType::DEVICE);
    }

    //! Provides a view of a section of data pointed to by the array
    //! Can also be thought of as one of the rows in the array
    //! but does not actually own the data - should be temporary
    struct view {
        //! Base pointer to index from (part of a larger array)
        T* base;
        //! Size of this section of memory
        std::size_t size;
        //! Steps per index
        std::size_t step_size;

        //! Provide an at operator into the memory (probably not used that much)
        GPU_COMPATIBLE
        T& at(
            std::size_t index
        ) const {
            #if not ON_GPU
                if(base == nullptr) throw std::runtime_error("base == nullptr, unable to index on CPU");
                if(index >= size) throw std::out_of_range("index is out of range in multi_array view");
            #endif

            return *(base + index * step_size);
        }

        //! Provide an index operator into the memory
        GPU_COMPATIBLE
        T& operator[](std::size_t index) const {
            #if not ON_GPU
                if(base == nullptr) throw std::runtime_error("base == nullptr, unable to index on CPU");
                if(index >= size) throw std::out_of_range("index is out of range in multi_array view");
            #endif

            return *(base + index * step_size);
        }
    }; /* struct view */

    /**
     * @brief Construct an empty host multi_array
     */
    multi_array():
        primary_location_(MemType::HOST),
        data_(nullptr),
        device_data_(nullptr),
        size_x_(0),
        size_y_(0)
    {}

    /**
     * @brief Construct a new shared multi_array object from an initializer list
     * @param data members to initialize data from
     * @param location what memory to create this multi_array in
     */
    multi_array(
        const std::initializer_list<std::initializer_list<T>>& data,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_x_(data.size()),
        size_y_(0)
    {
        if(size_x_ > 0) {
            std::size_t size = 0;
            bool set = false;

            for(auto& arr : data) {
                if(!set) {
                    size = arr.size();
                    set = true;
                }
                if(arr.size() > size) size = arr.size();
            }
            size_y_ = size;
        }

        allocate();

        if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                std::size_t i = 0;
                for(auto& arr : data) {
                    std::size_t j = 0;
                    for(auto& obj : arr) {
                        cudaMemcpy(device_data_ + i * size_y_ + j, &obj, sizeof(T), cudaMemcpyHostToDevice);
                        ++j;
                    }
                    ++i;
                }
            #endif
        } else {
            std::size_t i = 0;
            for(auto& arr : data) {
                std::size_t j = 0;
                for(auto& obj : arr) {
                    new(&data_[i * size_y_ + j]) T(obj);
                    ++j;
                }
                ++i;
            }
        }
    }
    
    /**
     * @brief Construct a new shared multi_array object with a fixed size, no construction of objects
     * @param size_x of the multi_array
     * @param size_y of the multi_array
     * @param location what memory to create this multi_array in
     */
    multi_array(
        std::size_t size_x,
        std::size_t size_y,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_x_(size_x),
        size_y_(size_y)
    {
        allocate();

        // No construction / initialization to do
    }
    
    /**
     * @brief Construct a new shared multi_array object of a size with a default value
     * @param size number of elements in multi_array
     * @param default_value the default value to initialize all elements from
     * @param location where to construct the multi_array
     */
    multi_array(
        std::size_t size_x,
        std::size_t size_y,
        const T& default_value,
        MemType location = MemType::HOST
    ):
        primary_location_(location),
        data_(nullptr),
        device_data_(nullptr),
        size_x_(size_x),
        size_y_(size_y)
    {
        allocate();

        // Mixed messaging over whether this is the right move
        // https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
        if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                for(std::size_t i = 0; i < size_x_; ++i) {
                    for(std::size_t j = 0; j < size_y_; ++j) {
                        cudaMemcpy(device_data_ + i * size_y_ + j, &default_value, sizeof(T), cudaMemcpyHostToDevice);
                    }
                }
            #endif
        } else {
            for(std::size_t i = 0; i < size_x_; ++i) {
                for(std::size_t j = 0; j < size_y_; ++j) {
                    new(&data_[i * size_y_ + j]) T(default_value);
                }
            }
        }
    }

    /**
     * @brief Construct a new shared multi_array object via a copy constructor
     * @param obj object to copy from
     */
    multi_array(const multi_array& obj):
        primary_location_(obj.primary_location_),
        data_(nullptr),
        device_data_(nullptr),
        size_x_(obj.size_x_),
        size_y_(obj.size_y_)
    {
        allocate();

        if(primary_location_ != MemType::DEVICE) {
            for(std::size_t i = 0; i < size_x_; ++i) {
                for(std::size_t j = 0; j < size_y_; ++j) {
                    new(&data_[i * size_y_ + j]) T(obj.data_[i * size_y_ + j]);
                }
            }
        } else {
            // Mixed messaging over whether this is the right move
            // https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
            #if CUDA_AVAILABLE
                cudaMemcpy(device_data_, obj.device_data_, sizeof(T) * size_x_ * size_y_, cudaMemcpyDeviceToDevice);
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
     * @brief Construct a new shared multi_array object via a copy constructor
     * @param obj object to copy from
     */
    multi_array(multi_array&& obj):
        primary_location_(obj.primary_location_),
        data_(obj.data_),
        device_data_(obj.device_data_),
        size_x_(obj.size_x_),
        size_y_(obj.size_y_)
    {
        obj.primary_location_ = MemType::HOST;
        obj.data_ = nullptr;
        obj.device_data_ = nullptr;
        obj.size_x_ = 0;
        obj.size_y_ = 0;
    }
    
    multi_array& operator=(const multi_array& obj)
    {
        deallocate();
        primary_location_ = obj.primary_location_;
        data_ = nullptr;
        device_data_ = nullptr;
        size_x_ = obj.size_x_;
        size_y_ = obj.size_y_;
        allocate();

        if(primary_location_ != MemType::DEVICE) {
            for(std::size_t i = 0; i < size_x_; ++i) {
                for(std::size_t j = 0; j < size_y_; ++j) {
                    new(&data_[i * size_y_ + j]) T(obj.data_[i * size_y_ + j]);
                }
            }
        } else {
            // THIS IS INEFFICIENT AF, but I'm lazy :)
            // eventually make a kernel call that does a byte for byte copy
            #if CUDA_AVAILABLE
                cudaMemcpy(device_data_, obj.device_data_, sizeof(T) * size_x_ * size_y_, cudaMemcpyDeviceToDevice);
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

    multi_array& operator=(multi_array&& obj) {
        primary_location_ = obj.primary_location_;
        data_ = obj.data_;
        device_data_ = obj.device_data_;
        size_x_ = obj.size_x_;
        size_y_ = obj.size_y_;

        obj.primary_location_ = MemType::HOST;
        obj.data_ = nullptr;
        obj.device_data_ = nullptr;
        obj.size_x_ = 0;
        obj.size_y_ = 0;
        return *this;
    }

    /**
     * @brief Destroy the shared multi_array object, deallocate any memory
     */
    ~multi_array() {
        deallocate();
    }

    /**
     * @brief get the size of the multi_array
     * @return std::size_t number of elements in multi_array
     */
    GPU_COMPATIBLE
    std::size_t size() const {
        return size_x_ * size_y_;
    }

    /**
     * @brief get the size of the multi_array along a particular dimension
     * @return std::size_t number of elements in multi_array
     */
    GPU_COMPATIBLE
    std::size_t size(std::size_t dim) const {
        if(dim == 0) return size_x_;
        if(dim == 1) return size_y_;
        return 0;
    }

    /**
     * @brief get the size of the multi_array (primary dimension)
     * @return std::size_t number of primary multi_arrays
     */
    GPU_COMPATIBLE
    std::size_t size_x() const {
        return size_x_;
    }

    /**
     * @brief get the size of the multi_array (secondary dimension)
     * @return std::size_t number of elements in multi_array
     */
    GPU_COMPATIBLE
    std::size_t size_y() const {
        return size_y_;
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
    T& at(
        std::size_t index_x,
        std::size_t index_y
    ) const {
        #if ON_GPU
            return device_data_[index_x + index_y * size_x_];
        #else
            if(data_ == nullptr) throw std::runtime_error("data_ == nullptr, unable to index on CPU");
            if(index_x >= size_x_ || index_y >= size_y_) throw std::out_of_range("index is out of range in multi_array");
            return data_[index_x + index_y * size_x_];
        #endif
    }



    GPU_COMPATIBLE
    view operator[](std::size_t index) const {
        #if ON_GPU
            view result;
            result.base = device_data_ + index;
            result.size = size_y_;
            result.step_size = size_x_;
            return result;
        #else
            if(data_ == nullptr) throw std::runtime_error("data_ == nullptr, unable to index on CPU");
            if(index >= size_x_) throw std::out_of_range("index is out of range in multi_array");

            view result;
            result.base = data_ + index;
            result.size = size_y_;
            result.step_size = size_x_;
            return result;
        #endif
    }


    /**
     * @brief transfer data from HOST to DEVICE (no effect if UNIFIED)
     */
    void fromGPU() {
        #if CUDA_AVAILABLE
            if (primary_location_ == MemType::DEVICE) {
                if(data_ == nullptr) {
                    data_ = static_cast<T*>(malloc(sizeof(T) * size_x_ * size_y_));
                }

                cudaMemcpy(data_, device_data_, sizeof(T) * size_x_ * size_y_, cudaMemcpyDeviceToHost);
            } else if(primary_location_ == MemType::HOST) {
                if(device_data_ == nullptr) {
                    throw std::runtime_error("DEVICE memory is not allocated, unable to copy from GPU");
                }
                if constexpr(multi_array_helpers::fromGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_x_; ++i) {
                        for(std::size_t j = 0; j < size_y_; ++j) {
                            bool copy = true;
                            copy = data_[i * size_y_ + j]->fromGPU(device_data_ + i * size_y_ + j);
                            if(copy)
                                cudaMemcpy(&data_[i * size_y_ + j], device_data_ + i * size_y_ + j, sizeof(T), cudaMemcpyDeviceToHost);
                        }
                    }
                } else {
                    if constexpr(multi_array_helpers::fromGPUVoidDefined<T>{}) {
                        for(std::size_t i = 0; i < size_x_; ++i) {
                            for(std::size_t j = 0; j < size_y_; ++j) {
                                data_[i * size_y_ + j]->fromGPU();
                            }
                        }
                    }
                    cudaMemcpy(data_, device_data_, sizeof(T) * size_x_ * size_y_, cudaMemcpyDeviceToHost);
                }
            } else if(primary_location_ == MemType::UNIFIED) {
                if constexpr(multi_array_helpers::fromGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_x_; ++i)
                        for(std::size_t j = 0; j < size_y_; ++j)
                            data_[i * size_y_ + j].fromGPU(nullptr);
                } else if constexpr(multi_array_helpers::fromGPUVoidDefined<T>{}) {
                    for(std::size_t i = 0; i < size_x_; ++i)
                        for(std::size_t j = 0; j < size_y_; ++j)
                            data_[i * size_y_ + j].fromGPU();
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

                cudaMemcpy(device_data_, data_, sizeof(T) * size_x_ * size_y_, cudaMemcpyHostToDevice);
            } else if(primary_location_ == MemType::HOST) {
                if(device_data_ == nullptr) {
                    cudaMalloc(&device_data_, sizeof(T) * size_x_ * size_y_);
                }
                if constexpr(multi_array_helpers::toGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_x_; ++i) {
                        for(std::size_t j = 0; j < size_y_; ++j) {
                            bool copy = true;
                            copy = data_[i * size_y_ + j]->toGPU(device_data_ + i * size_y_ + j);
                            if(copy)
                                cudaMemcpy(device_data_ + i * size_y_ + j, &data_[i * size_y_ + j], sizeof(T), cudaMemcpyHostToDevice);
                        }
                    }
                } else {
                    if constexpr(multi_array_helpers::toGPUVoidDefined<T>{}) {
                        for(std::size_t i = 0; i < size_x_; ++i) {
                            for(std::size_t j = 0; j < size_y_; ++j) {
                                data_[i * size_y_ + j]->toGPU();
                            }
                        }
                    }
                    cudaMemcpy(device_data_, data_, sizeof(T) * size_x_ * size_y_, cudaMemcpyHostToDevice);
                }
            } else if(primary_location_ == MemType::UNIFIED) {
                if constexpr(multi_array_helpers::toGPUDefined<T>{}) {
                    for(std::size_t i = 0; i < size_x_; ++i)
                        for(std::size_t j = 0; j < size_y_; ++j)
                            data_[i * size_y_ + j].toGPU(nullptr);
                } else if constexpr(multi_array_helpers::toGPUVoidDefined<T>{}) {
                    for(std::size_t i = 0; i < size_x_; ++i)
                        for(std::size_t j = 0; j < size_y_; ++j)
                            data_[i * size_y_ + j].toGPU();
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
            for(std::size_t i = 0; i < size_x_; ++i) {
                for(std::size_t j = 0; j < size_y_; ++j) {
                    data_[i * size_y_ + j].~T();
                }
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
                std::fprintf(stderr, "ERROR: de-allocating unified multi_array, but CUDA is not available, ensure full project is compiled with CUDA.\n");
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
        auto dim = size_x_ * size_y_;
        if(dim == 0) return;
        if(primary_location_ == MemType::HOST) {
            data_ = static_cast<T*>(malloc(sizeof(T) * size_x_ * size_y_));
        } else if(primary_location_ == MemType::UNIFIED) {
            #if CUDA_AVAILABLE
                cudaMallocManaged(&data_, sizeof(T) * size_x_ * size_y_);
                device_data_ = data_;
            #else
                throw std::runtime_error("CUDA Not available, unable to allocated unified memory");
            #endif
        } else if(primary_location_ == MemType::DEVICE) {
            #if CUDA_AVAILABLE
                cudaMalloc(&device_data_, sizeof(T) * size_x_ * size_y_);
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
    //! Size of the array (x dim)
    std::size_t size_x_;
    //! Size of the array (y dim)
    std::size_t size_y_;

}; /* class multi_array */

} /* namespace jump */

#endif /* JUMP_MULTI_ARRAY_HPP_ */