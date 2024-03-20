/**
 * @file array.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief tests for the jump::array class
 * @date 2023-02-09
 */
#include <jump/array.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME Array
#else
    #define TEST_SUITE_NAME ArrayNoCuda
#endif

namespace array_test {

template<typename T>
struct SizeCheck {
    using TestResult = std::size_t;
    using Arguments = jump::array<T>;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size();
    }
};

template<typename T>
struct SetValues {
    struct Arguments {
        jump::array<T> arr;
        T start_value;
    };
    using TestResult = bool;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& aa = *const_cast<Arguments*>(&args);
        for(int i = 0; i < aa.arr.size(); ++i)
            aa.arr.at(i) = aa.start_value + i;
    }
};

template<typename T>
struct CheckValues {
    struct Arguments {
        jump::array<T> arr;
        T start_value;
    };
    using TestResult = bool;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& aa = *const_cast<Arguments*>(&args);
        for(int i = 0; i < aa.arr.size(); ++i) {
            if(aa.arr.at(i) != aa.start_value + i) {
                result = false;
                return;
            }
        }
        result = true;
    }
};

// TODO(jspisak): create class that can record construction / descruction events
//  and make sure that the array is handling this correctly

} /* namespace array_test */

TEST(TEST_SUITE_NAME, sizeOnlyConstructorHOST) {
    jump::array<int> arr(10, jump::memory_t::HOST);
    ASSERT_EQ(arr.size(), 10);
}

TEST(TEST_SUITE_NAME, sizeAndDefaultConstructorHOST) {
    jump::array<int> arr(10, 11, jump::memory_t::HOST);
    ASSERT_EQ(arr.size(), 10);
    for(int i = 0; i < 10; ++i)
        ASSERT_EQ(arr.at(i), 11);
}

TEST(TEST_SUITE_NAME, valsConstructorHOST) {
    jump::array<int> arr({1, 2, 3, 4, 5}, jump::memory_t::HOST);
    ASSERT_EQ(arr.size(), 5);
    for(int i = 0; i < 5; ++i)
        ASSERT_EQ(arr.at(i), i + 1);
}

TEST(TEST_SUITE_NAME, sizeOnlyConstructorUNIFIED) {
    bool exception = false;
    try {
        jump::array<int> arr(10, jump::memory_t::UNIFIED);
        ASSERT_EQ(arr.size(), 10);
    } catch(std::exception& e) {
        exception = true;
    }
    
    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, sizeAndDefaultConstructorUNIFIED) {
    bool exception = false;
    try {
        jump::array<int> arr(10, 11, jump::memory_t::UNIFIED);
        ASSERT_EQ(arr.size(), 10);
    for(int i = 0; i < 10; ++i)
        ASSERT_EQ(arr.at(i), 11);
    } catch(std::exception& e) {
        exception = true;
    }
    
    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, valsConstructorUNIFIED) {
    bool exception = false;
    try {
        jump::array<int> arr({1, 2, 3, 4, 5}, jump::memory_t::UNIFIED);
        ASSERT_EQ(arr.size(), 5);
        for(int i = 0; i < 5; ++i)
            ASSERT_EQ(arr.at(i), i + 1);
    } catch(std::exception& e) {
        exception = true;
    }
    
    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, sizeOnlyConstructorDEVICE) {
    bool exception = false;
    try {
        jump::array<int> arr(10, jump::memory_t::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, ValuePassingHostDevice) {
    jump::array<int> arr(10, jump::memory_t::HOST);

    using SetValRunner = jump_testing::InteropableTestRunner<array_test::SetValues<int>>;
    using CheckValRunner = jump_testing::InteropableTestRunner<array_test::CheckValues<int>>;

    SetValRunner sv_runner;
    array_test::SetValues<int>::Arguments sv_args;
    sv_args.arr = arr;
    sv_args.start_value = 0;
    sv_runner.host(sv_args);

    CheckValRunner cv_runner;
    array_test::CheckValues<int>::Arguments cv_args;
    cv_args.arr = arr;
    cv_args.start_value = 0;
    ASSERT_TRUE(cv_runner.host(cv_args));

    bool exception = false;
    try {
        arr.to_device();
        cv_args.arr.sync();
        ASSERT_TRUE(cv_runner.device(cv_args));
        ASSERT_TRUE(cv_runner.device_copy(cv_args));
        sv_args.arr.sync();
        sv_args.start_value = 1;
        cv_args.start_value = 1;
        sv_runner.device(sv_args);
        ASSERT_TRUE(cv_runner.device(cv_args));
        ASSERT_TRUE(cv_runner.device_copy(cv_args));
        ASSERT_FALSE(cv_runner.host(cv_args));
        arr.from_device();
        ASSERT_TRUE(cv_runner.host(cv_args));
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}


TEST(TEST_SUITE_NAME, ValuePassingUnified) {
    bool exception = false;
    try {
        jump::array<int> arr(10, jump::memory_t::UNIFIED);

        using SetValRunner = jump_testing::InteropableTestRunner<array_test::SetValues<int>>;
        using CheckValRunner = jump_testing::InteropableTestRunner<array_test::CheckValues<int>>;
        SetValRunner sv_runner;
        array_test::SetValues<int>::Arguments sv_args;
        sv_args.arr = arr;
        sv_args.start_value = 0;
        sv_runner.host(sv_args);

        CheckValRunner cv_runner;
        array_test::CheckValues<int>::Arguments cv_args;
        cv_args.arr = arr;
        cv_args.start_value = 0;
        ASSERT_TRUE(cv_runner.host(cv_args));
        ASSERT_TRUE(cv_runner.device(cv_args));
        ASSERT_TRUE(cv_runner.device_copy(cv_args));
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, arraySizeTest) {
    jump::array<int> arr(10, jump::memory_t::HOST);

    using SizeCheckRunner = jump_testing::InteropableTestRunner<array_test::SizeCheck<int>>;
    SizeCheckRunner sc_runner;
    ASSERT_EQ(sc_runner.host(arr), 10);

    bool exception = false;
    try {
        arr.to_device();
        ASSERT_EQ(sc_runner.device(arr), 10);
        ASSERT_EQ(sc_runner.device_copy(arr), 10);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, outOfBoundsException) {
    jump::array<int> arr(10, 10, jump::memory_t::HOST);

    ASSERT_EQ(arr.at(9), 10);

    bool exception = false;
    try {
        ASSERT_EQ(arr.at(10), 10);
    } catch(std::exception& e) {
        exception = true;
    }

    ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, atIndexEquivalence) {
    jump::array<int> arr({1, 2, 3, 4, 6, 7, 10}, jump::memory_t::HOST);

    for(int i = 0; i < arr.size(); ++i)
        ASSERT_EQ(arr[i], arr.at(i));
}

TEST(TEST_SUITE_NAME, capacity) {
    jump::array<int> arr(10, 15, jump::memory_t::HOST);
    arr.reserve(arr.capacity() + 3);
    ASSERT_EQ(arr.size(), 10);
    ASSERT_EQ(arr.capacity(), 13);

    for(int i = 0; i < arr.size(); ++i)
        ASSERT_EQ(arr[i], 15);
}

TEST(TEST_SUITE_NAME, pushBack) {
    jump::array<int> arr(10, 15, jump::memory_t::HOST);
    arr.reserve(arr.capacity() + 3);
    ASSERT_EQ(arr.size(), 10);
    ASSERT_EQ(arr.capacity(), 13);

    for(int i = 0; i < arr.size(); ++i)
        ASSERT_EQ(arr[i], 15);

    arr.push_back(3);
    arr.push_back(4);
    arr.push_back(5);
    ASSERT_EQ(arr.size(), 13);
    ASSERT_EQ(arr.capacity(), 13);
    arr.push_back(6);
    for(int i = 0; i < 10; ++i)
        ASSERT_EQ(arr[i], 15);
    ASSERT_EQ(arr.size(), 14);
    ASSERT_EQ(arr.capacity(), 100);
}

TEST(TEST_SUITE_NAME, commaOperator) {
    jump::array<int> arr(5);
    arr << 1, 2, 3, 4, 5;
    ASSERT_EQ(arr[0], 1);
    ASSERT_EQ(arr[1], 2);
    ASSERT_EQ(arr[2], 3);
    ASSERT_EQ(arr[3], 4);
    ASSERT_EQ(arr[4], 5);

    bool exception = false;
    try {
        arr << 1, 2, 3, 4, 5, 6;
    } catch(const std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, iterator) {
    jump::array<int> arr(5);
    arr << 1, 2, 3, 4, 5;

    int value = 1;
    for(const auto& v : arr) {
        ASSERT_EQ(value, v);
        ++value;
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
