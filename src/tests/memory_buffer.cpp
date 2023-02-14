/**
 * @file memory_buffer.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief tests for the jump::memory_buffer class
 * @date 2022-10-06
 */
#include <jump/memory_buffer.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME MemoryBuffer
#else
    #define TEST_SUITE_NAME MemoryBufferNoCuda
#endif

namespace memory_buffer_test {

struct SizeCheck {
    using TestResult = std::size_t;
    using Arguments = jump::memory_buffer;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size();
    }
};

struct GetSetValue {
    struct Arguments {
        jump::memory_buffer buffer;
        int value;
    };
    using TestResult = int;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.buffer.data<int>()[0];
        args.buffer.data<int>()[0] = args.value;
    }
};

} /* namespace memory_buffer_test */

TEST(TEST_SUITE_NAME, emptyConstructor) {
    jump::memory_buffer empty_buffer;
    ASSERT_EQ(empty_buffer.data(), nullptr);
    ASSERT_EQ(empty_buffer.host_data(), nullptr);
    ASSERT_EQ(empty_buffer.device_data(), nullptr);
    ASSERT_EQ(empty_buffer.data<int>(), nullptr);
    ASSERT_EQ(empty_buffer.host_data<int>(), nullptr);
    ASSERT_EQ(empty_buffer.device_data<int>(), nullptr);
}

TEST(TEST_SUITE_NAME, deallocate) {
    jump::memory_buffer buf;
    buf.allocate(10, jump::memory_t::HOST);
    ASSERT_FALSE(buf.data() == nullptr);
    ASSERT_TRUE(buf.data() == buf.host_data());
    ASSERT_TRUE(buf.device_data() == nullptr);
    ASSERT_EQ(buf.size(), 10);
    buf.deallocate();
    ASSERT_EQ(buf.data(), nullptr);
    ASSERT_EQ(buf.host_data(), nullptr);
    ASSERT_EQ(buf.device_data(), nullptr);
}

TEST(TEST_SUITE_NAME, allocateVoidHost) {
    jump::memory_buffer buf;
    buf.allocate(10, jump::memory_t::HOST);
    ASSERT_FALSE(buf.data() == nullptr);
    ASSERT_TRUE(buf.data() == buf.host_data());
    ASSERT_TRUE(buf.device_data() == nullptr);
}

TEST(TEST_SUITE_NAME, allocateVoidDevice) {
    jump::memory_buffer buf;

    bool exception = false;
    try {
        buf.allocate(10, jump::memory_t::DEVICE);
        ASSERT_TRUE(buf.data() == nullptr);
        ASSERT_FALSE(buf.device_data() == nullptr);
    } catch(std::exception& e) {
        exception = true;
    }
    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, allocateVoidUnified) {
    jump::memory_buffer buf;

    bool exception = false;
    try {
        buf.allocate(10, jump::memory_t::UNIFIED);
        ASSERT_FALSE(buf.data() == nullptr);
        ASSERT_FALSE(buf.device_data() == nullptr);
        ASSERT_EQ(buf.data(), buf.device_data());
    } catch(std::exception& e) {
        exception = true;
    }
    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, allocateIntHost) {
    jump::memory_buffer buf;
    buf.allocate<int>(10, jump::memory_t::HOST);
    ASSERT_EQ(buf.size(), sizeof(int) * 10);
}

TEST(TEST_SUITE_NAME, copyConstruct) {
    jump::memory_buffer buf;
    jump::memory_buffer buf2(buf);
    bool exception = false;
    try {
        buf.allocate<int>(10, jump::memory_t::UNIFIED);
        ASSERT_EQ(buf.data(), buf2.data());
        ASSERT_EQ(buf.device_data(), buf2.device_data());
        ASSERT_EQ(buf.size(), buf2.size());
        ASSERT_EQ(buf.owners(), 2);
    } catch(std::exception& e) {
        exception = true;
    }
    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, moveConstruct) {
    jump::memory_buffer buf;

    bool exception = false;
    try {
        buf.allocate(10, jump::memory_t::UNIFIED);
        ASSERT_EQ(buf.size(), 10);
        ASSERT_FALSE(buf.data() == nullptr);
        jump::memory_buffer buf2(std::move(buf));
        ASSERT_TRUE(buf.block() == nullptr);
        ASSERT_EQ(buf2.size(), 10);
        ASSERT_FALSE(buf2.data() == nullptr);
        ASSERT_EQ(buf2.owners(), 1);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, copyAssign) {
    jump::memory_buffer buf;
    jump::memory_buffer buf2 = buf;

    bool exception = false;
    try {
        buf.allocate<int>(10, jump::memory_t::UNIFIED);
        ASSERT_EQ(buf.data(), buf2.data());
        ASSERT_EQ(buf.device_data(), buf2.device_data());
        ASSERT_EQ(buf.size(), buf2.size());
        ASSERT_EQ(buf.owners(), 2);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, moveAssign) {
    jump::memory_buffer buf;

    bool exception = false;
    try {
        buf.allocate(10, jump::memory_t::UNIFIED);
        ASSERT_EQ(buf.size(), 10);
        ASSERT_FALSE(buf.data() == nullptr);
        jump::memory_buffer buf2 = std::move(buf);
        ASSERT_TRUE(buf.block() == nullptr);
        ASSERT_EQ(buf2.size(), 10);
        ASSERT_FALSE(buf2.data() == nullptr);
        ASSERT_EQ(buf2.owners(), 1);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, hostToDevice) {
    jump::memory_buffer buf;
    buf.allocate(10, jump::memory_t::HOST);
    ASSERT_TRUE(buf.device_data() == nullptr);

    bool exception = false;
    try {
        buf.to_device();
        ASSERT_FALSE(buf.device_data() == nullptr);
        buf.from_device();
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, hostDeviceToHost) {
    jump::memory_buffer buf;
    buf.allocate(10, jump::memory_t::HOST);
    ASSERT_TRUE(buf.device_data() == nullptr);
    ASSERT_FALSE(buf.host_data() == nullptr);
    bool exception = false;
    try {
        buf.from_device();
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, deviceToHost) {
    jump::memory_buffer buf;

    bool exception = false;
    try {
        buf.allocate(10, jump::memory_t::DEVICE);
        ASSERT_TRUE(buf.host_data() == nullptr);
        buf.from_device();
        ASSERT_FALSE(buf.host_data() == nullptr);
        buf.to_device();
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, deviceHostToDevice) {
    jump::memory_buffer buf;
    
    bool upper_exception = false;
    try {
        buf.allocate(10, jump::memory_t::DEVICE);
        ASSERT_TRUE(buf.host_data() == nullptr);
        ASSERT_FALSE(buf.device_data() == nullptr);
        bool exception = false;
        try {
            buf.to_device();
        } catch(std::exception& e) {
            exception = true;
        }
        ASSERT_TRUE(exception);
    
    } catch(std::exception& e) {
        upper_exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(upper_exception);
    } else {
        ASSERT_TRUE(upper_exception);
    }
}

TEST(TEST_SUITE_NAME, sizeTest) {
    using Test = jump_testing::InteropableTestRunner<memory_buffer_test::SizeCheck>;
    Test test;

    jump::memory_buffer buf;
    bool exception = false;
    try {
        buf.allocate(20, jump::memory_t::UNIFIED);
        ASSERT_EQ(test.host(buf), 20);
        ASSERT_EQ(test.device(buf), 20);
        ASSERT_EQ(test.device_copy(buf), 20);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, valueSettingUnified) {
    using Test = jump_testing::InteropableTestRunner<memory_buffer_test::GetSetValue>;
    memory_buffer_test::GetSetValue::Arguments args;

    bool exception = false;
    try {
        args.buffer.allocate(20, jump::memory_t::UNIFIED);
        args.value = 10;

        Test test;
        test.host(args);
        args.value = 30;
        ASSERT_EQ(test.host(args), 10);
        args.value = 20;
        ASSERT_EQ(test.device(args), 30);
        args.value = 10;
        ASSERT_EQ(test.device_copy(args), 20);
        ASSERT_EQ(test.host(args), 10);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, valueSettingHostDevice) {
    using Test = jump_testing::InteropableTestRunner<memory_buffer_test::GetSetValue>;
    memory_buffer_test::GetSetValue::Arguments args;
    args.buffer.allocate(20, jump::memory_t::HOST);
    args.value = 10;

    Test test;
    test.host(args);
    args.value = 30;
    ASSERT_EQ(test.host(args), 10);
    ASSERT_EQ(test.host(args), 30);

    bool exception = false;
    try {
        args.buffer.to_device();
        args.value = 20;
        ASSERT_EQ(test.device(args), 30);
        args.value = 10;
        ASSERT_EQ(test.device_copy(args), 20);
        ASSERT_EQ(test.host(args), 30);
        args.buffer.from_device();
        ASSERT_EQ(test.host(args), 10);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, valueSettingDeviceHost) {
    using Test = jump_testing::InteropableTestRunner<memory_buffer_test::GetSetValue>;
    memory_buffer_test::GetSetValue::Arguments args;

    bool exception = false;
    try {
        args.buffer.allocate(20, jump::memory_t::DEVICE);
        args.value = 10;

        Test test;
        test.device(args);
        args.value = 30;
        ASSERT_EQ(test.device(args), 10);
        args.buffer.from_device();
        args.value = 20;
        ASSERT_EQ(test.host(args), 30);
        ASSERT_EQ(test.device_copy(args), 30);
        args.buffer.to_device();
        args.value = 10;
        ASSERT_EQ(test.device(args), 20);
        args.buffer.from_device();
        ASSERT_EQ(test.host(args), 10);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, reallocateHOST) {
    jump::memory_buffer buf;
    buf.allocate<int>(10, jump::memory_t::HOST);
    buf.data<int>()[9] = 2;
    ASSERT_EQ(buf.size(), 10 * sizeof(int));
    ASSERT_EQ(buf.data<int>()[9], 2);
    buf.reallocate<int>(20);
    ASSERT_EQ(buf.size(), 20 * sizeof(int));
    ASSERT_EQ(buf.data<int>()[9], 2);
}

TEST(TEST_SUITE_NAME, reallocateUNIFIED) {
    jump::memory_buffer buf;

    auto exception = false;
    try {
        buf.allocate<int>(10, jump::memory_t::UNIFIED);
        buf.data<int>()[9] = 2;
        ASSERT_EQ(buf.size(), 10 * sizeof(int));
        ASSERT_EQ(buf.data<int>()[9], 2);
        buf.reallocate<int>(20);
        ASSERT_EQ(buf.size(), 20 * sizeof(int));
        ASSERT_EQ(buf.data<int>()[9], 2);
    } catch(std::exception& e) {
        exception = true;
        if constexpr(jump::cuda_enabled()) {
            throw e;
        }
    }
    if constexpr(!jump::cuda_enabled()) {
        ASSERT_TRUE(exception);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
