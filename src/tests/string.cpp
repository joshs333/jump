/**
 * @file string_view.cpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief tests for the jump::string_view class
 * @date 2024-02-06
 */
#include <jump/string.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME String
#else
    #define TEST_SUITE_NAME StringNoCuda
#endif

namespace string_test {

struct SizeCheck {
    using TestResult = std::size_t;
    using Arguments = jump::string;

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size();
    }
};

} /* namespace string_view_test */


TEST(TEST_SUITE_NAME, length) {
    using Test = jump_testing::InteropableTestRunner<string_test::SizeCheck>;
    Test test;

    jump::string a("hello");
    jump::string b("hello2");

    ASSERT_EQ(test.host(a), 5);
    ASSERT_EQ(test.host(b), 6);

    bool exception = false;
    try {
        a.to_device();
        b.to_device();
        ASSERT_EQ(test.device(a), 5);
        ASSERT_EQ(test.device_copy(a), 5);
        ASSERT_EQ(test.device(b), 6);
        ASSERT_EQ(test.device_copy(b), 6);
    } catch(std::exception& e) {
        exception = true;
    }

    if constexpr(jump::cuda_enabled()) {
        ASSERT_FALSE(exception);
    } else {
        ASSERT_TRUE(exception);
    }
}

TEST(TEST_SUITE_NAME, comparison) {
    jump::string a("hello");
    jump::string b("hello2");

    ASSERT_TRUE(a == a);
    ASSERT_TRUE(b == b);
    ASSERT_FALSE(a == b);
    ASSERT_FALSE(b == a);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
