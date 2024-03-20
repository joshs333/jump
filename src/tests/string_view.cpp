/**
 * @file string_view.cpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief tests for the jump::string_view class
 * @date 2024-02-06
 */
#include <jump/string_view.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME StringView
#else
    #define TEST_SUITE_NAME StringViewNoCuda
#endif

namespace string_view_test {

} /* namespace string_view_test */


TEST(TEST_SUITE_NAME, length) {
    ASSERT_EQ(jump::string_view("hello").len(), 5);
    ASSERT_EQ(jump::string_view("hello2").len(), 6);
}


TEST(TEST_SUITE_NAME, stringComparison) {
    ASSERT_TRUE(jump::string_view("hello") == jump::string("hello"));
    ASSERT_TRUE(jump::string_view("hello") != jump::string("hello2"));
    ASSERT_FALSE(jump::string_view("hello") == jump::string("hello2"));
    ASSERT_FALSE(jump::string_view("hello") != jump::string("hello"));
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
