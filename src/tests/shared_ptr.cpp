/**
 * @file memory_buffer.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief tests for the jump::memory_buffer class
 * @date 2022-10-06
 */
#include <jump/shared_ptr.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME SharedPtr
#else
    #define TEST_SUITE_NAME SharedPtrNoCuda
#endif

namespace shared_ptr_testing {

template<typename T>
struct GetSetValue {
    struct Arguments {
        jump::shared_ptr<T> ptr;
        T value;
    };
    using TestResult = T;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = *args.ptr;
        *args.ptr.get() = args.value;
    }
};

struct Functor {
    int value = 0;

    JUMP_INTEROPABLE
    int function() const {
        return value;
    }
};

struct FunctionCalling {
    using Arguments  = jump::shared_ptr<Functor>;
    using TestResult = int;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = args->function();
    }
};


} /* shared_ptr_testing */

TEST(TEST_SUITE_NAME, makeSharedOnHost) {
    auto ptr = jump::make_shared_on<int>(jump::memory_t::HOST, 10);
    ASSERT_EQ(*ptr, 10);
    ASSERT_EQ(*ptr.get(), 10);
}

TEST(TEST_SUITE_NAME, makeSharedOnUnified) {
    bool exception = false;

    try {
        auto ptr = jump::make_shared_on_unified<int>(10);
        ASSERT_EQ(*ptr, 10);
        ASSERT_EQ(*ptr.get(), 10);
    } catch(std::exception& e) {
        // if cuda is enabled, this shouldn't happen and let's throw
        if constexpr(jump::cuda_enabled())
            throw e;
        exception = true;
    }

    // if cuda is not enabled, we except the exception
    if constexpr(!jump::cuda_enabled())
        ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, valueTestingHostDevice) {
    using Test = shared_ptr_testing::GetSetValue<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;
    Test::Arguments args;

    auto ptr = jump::make_shared_on<int>(jump::memory_t::HOST, 0);
    ASSERT_EQ(*ptr, 0);

    args.ptr = ptr;
    args.value = 1;
    ASSERT_EQ(runner.host(args), 0);

    bool exception = false;
    try {
        ptr.to_device();
        args.ptr.sync();
        args.value = 2;
        ASSERT_EQ(runner.device(args), 1);
        args.value = 3;
        ASSERT_EQ(runner.device_copy(args), 2);
        ASSERT_EQ(runner.host(args), 1);
        ptr.from_device();
        ASSERT_EQ(runner.host(args), 3);
    } catch(std::exception& e) {
        if constexpr(jump::cuda_enabled())
            throw e;
        exception = true;
    }

    if constexpr(!jump::cuda_enabled())
        ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, valueTestingUnified) {
    using Test = shared_ptr_testing::GetSetValue<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;
    Test::Arguments args;

    bool exception = false;
    try {
        auto ptr = jump::make_shared_on_unified<int>(0);
        ASSERT_EQ(*ptr, 0);

        args.ptr = ptr;
        args.value = 1;
        ASSERT_EQ(runner.host(args), 0);
        ASSERT_EQ(runner.host(args), 1);
        args.value = 2;
        ASSERT_EQ(runner.device(args), 1);
        args.value = 3;
        ASSERT_EQ(runner.device_copy(args), 2);
        ASSERT_EQ(runner.host(args), 3);
    } catch(std::exception& e) {
        if constexpr(jump::cuda_enabled())
            throw e;
        exception = true;
    }

    if constexpr(!jump::cuda_enabled())
        ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, functionTestingHostDevice) {
    using Test = shared_ptr_testing::FunctionCalling;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    auto ptr = jump::make_shared_on_host<shared_ptr_testing::Functor>();
    ptr->value = 1;
    ASSERT_EQ(runner.host(ptr), 1);

    bool exception = false;
    try {
        auto ptr2 = ptr;
        ptr.to_device();
        ptr2.sync();
        ASSERT_EQ(runner.device(ptr2), 1);
        ASSERT_EQ(runner.device_copy(ptr2), 1);
    } catch(std::exception& e) {
        if constexpr(jump::cuda_enabled())
            throw e;
        exception = true;
    }

    if constexpr(!jump::cuda_enabled())
        ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, functionTestingUnified) {
    using Test = shared_ptr_testing::FunctionCalling;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    bool exception = false;
    try {
        auto ptr = jump::make_shared_on_unified<shared_ptr_testing::Functor>();
        auto ptr2 = ptr;
        ptr->value = 1;
        ASSERT_EQ(runner.host(ptr), 1);
        ptr->value = 2;
        ASSERT_EQ(runner.device(ptr2), 2);
        ptr2->value = 3;
        ASSERT_EQ(runner.device_copy(ptr), 3);
    } catch(std::exception& e) {
        if constexpr(jump::cuda_enabled())
            throw e;
        exception = true;
    }

    if constexpr(!jump::cuda_enabled())
        ASSERT_TRUE(exception);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
