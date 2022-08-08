#include <jump/device_interface.hpp>
#include <iostream>

struct TestA {
    static const bool host_compatible = false;
    static const bool device_compatible = true;

    void to_device() {

    }

    void from_device() {

    }
};


struct TestB {
    static const bool host_compatible = true;
    static const bool device_compatible = false;
};

int main() {
    std::cout << std::endl;
    std::cout << "Cuda available: " << jump::cuda_available() << std::endl;
    std::cout << "Device Count:   " << jump::cuda_device_count() << std::endl;
    std::cout << std::endl;

    std::cout << "Test A" << std::endl;
    std::cout << "Host Compat:   " << jump::kernel_interface<TestA>::host_compatible() << std::endl;
    std::cout << "Device Compat: " << jump::kernel_interface<TestA>::device_compatible() << std::endl;
    std::cout << "To Device:     " << jump::kernel_interface<TestA>::to_device_defined() << std::endl;
    std::cout << "From Device:   " << jump::kernel_interface<TestA>::from_device_defined() << std::endl;
    std::cout << std::endl;

    std::cout << "Test B" << std::endl;
    std::cout << "Host Compat:   " << jump::kernel_interface<TestB>::host_compatible() << std::endl;
    std::cout << "Device Compat: " << jump::kernel_interface<TestB>::device_compatible() << std::endl;
    std::cout << "To Device:     " << jump::kernel_interface<TestB>::to_device_defined() << std::endl;
    std::cout << "From Device:   " << jump::kernel_interface<TestB>::from_device_defined() << std::endl;

    #ifdef JUMP_ENABLE_CUDA
        std::printf("Cuda enabled at compile time.\n");
        // if(jump::cuda_available()) {
            std::printf("Attempting to allocate!\n");
            void* data_ptr = nullptr;
            auto r = cudaMalloc(&data_ptr, 100);
            std::cout << r << std::endl;
            std::printf("Done!\n");
        // }
    #endif
}
