/**
 * @file atomic.cpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief tests for the jump::atomic class
 * @date 2024-02-20
 * @note because the CUDA atomic operations do not
 *  work on local memory, we don't run the test runner
 *  device test (which copies the atomic to local memory
 *  as a bare argument to the kernel)
 */
#include <jump/atomic.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME Atomic
#else
    #define TEST_SUITE_NAME AtomicNoCuda
#endif

namespace atomic_test {

template<typename T>
struct ValuePassthrough {
    using Arguments = jump::atomic<T>;
    using TestResult = T;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& a = *const_cast<Arguments*>(&args);
        a = a;
        result = a;
    }
};

template<typename T>
struct ValueSet {
    struct Arguments {
        jump::atomic<T> atomic;
        T value;
    };
    using TestResult = T;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& a = *const_cast<jump::atomic<T>*>(&(args.atomic));
        a = args.value;
        result = a;
    }
};

template<typename T>
struct AtomicPassthrough {
    using Arguments = jump::atomic<T>;
    using TestResult = jump::atomic<T>;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = args;
    }
};

template<typename T>
struct AdditionTest {
    using Arguments = jump::atomic<T>;
    struct TestResult {
        T a;
        T b;
        T c;
        T d;
    };

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& a = *const_cast<jump::atomic<T>*>(&args);
        result.a = a++; // a
        result.b = ++a; // a + 2
        a += 7;
        result.c = a.fetch_add(7); // a + 9
        result.d = a; // a + 16
    }
};

template<typename T>
struct SubtractionTest {
    using Arguments = jump::atomic<T>;
    struct TestResult {
        T a;
        T b;
        T c;
        T d;
    };

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& a = *const_cast<jump::atomic<T>*>(&args);
        result.a = a--; // a
        result.b = --a; // a - 2
        a -= 7;
        result.c = a.fetch_subtract(7); // a - 9
        result.d = a; // a - 16
    }
};

template<typename T>
struct MinimumTest {
    using Arguments = jump::atomic<T>;
    using TestResult = T;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& a = *const_cast<jump::atomic<T>*>(&args);
        a.fetch_min(100);
        result = a;
    }
};

template<typename T>
struct MaximumTest {
    using Arguments = jump::atomic<T>;
    using TestResult = T;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        auto& a = *const_cast<jump::atomic<T>*>(&args);
        a.fetch_max(100);
        result = a;
    }
};

} /* namespace atomic_test */

TEST(TEST_SUITE_NAME, ValuePassthrough) {
    using Test = atomic_test::ValuePassthrough<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    ASSERT_EQ(runner.host(5), 5);
    ASSERT_EQ(runner.device_copy(5), 5);
}

TEST(TEST_SUITE_NAME, ValueSet) {
    using Test = atomic_test::ValueSet<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    ASSERT_EQ(runner.host({2, 5}), 5);
    ASSERT_EQ(runner.device_copy({2, 5}), 5);
}

TEST(TEST_SUITE_NAME, AtomicPassthrough) {
    using Test = atomic_test::AtomicPassthrough<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    ASSERT_EQ(runner.host(5), 5);
    ASSERT_EQ(runner.device_copy(5), 5);
}

TEST(TEST_SUITE_NAME, AdditionTest) {
    using Test = atomic_test::AdditionTest<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    ASSERT_EQ(runner.host(0).a, 0);
    ASSERT_EQ(runner.host(0).b, 2);
    ASSERT_EQ(runner.host(0).c, 9);
    ASSERT_EQ(runner.host(0).d, 16);
    ASSERT_EQ(runner.device_copy(0).a, 0);
    ASSERT_EQ(runner.device_copy(0).b, 2);
    ASSERT_EQ(runner.device_copy(0).c, 9);
    ASSERT_EQ(runner.device_copy(0).d, 16);
}

TEST(TEST_SUITE_NAME, SubtractionTest) {
    using Test = atomic_test::SubtractionTest<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    ASSERT_EQ(runner.host(16).a, 16);
    ASSERT_EQ(runner.host(16).b, 14);
    ASSERT_EQ(runner.host(16).c, 7);
    ASSERT_EQ(runner.host(16).d, 0);
    ASSERT_EQ(runner.device_copy(16).a, 16);
    ASSERT_EQ(runner.device_copy(16).b, 14);
    ASSERT_EQ(runner.device_copy(16).c, 7);
    ASSERT_EQ(runner.device_copy(16).d, 0);
}


TEST(TEST_SUITE_NAME, MinimumTest) {
    using Test = atomic_test::MinimumTest<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;

    ASSERT_EQ(runner.host(200), 100);
    ASSERT_EQ(runner.host(1), 1);
}


TEST(TEST_SUITE_NAME, MaximumTest) {
    using Test = atomic_test::MaximumTest<int>;
    using TestRunner = jump_testing::InteropableTestRunner<Test>;
    TestRunner runner;
    ASSERT_EQ(runner.host(200), 200);
    ASSERT_EQ(runner.host(1), 100);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
