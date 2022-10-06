/**
 * @file memory_buffer.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief tests for the jump::memory_buffer class
 * @date 2022-10-06
 */
#include <jump/memory_buffer.hpp>

#include <gtest/gtest.h>

TEST(MemoryBuffer, emptyConstructor) {
    jump::memory_buffer empty_buffer;
    ASSERT_EQ(empty_buffer.data(), nullptr);
    ASSERT_EQ(empty_buffer.host_data(), nullptr);
    ASSERT_EQ(empty_buffer.device_data(), nullptr);
    ASSERT_EQ(empty_buffer.data<int>(), nullptr);
    ASSERT_EQ(empty_buffer.host_data<int>(), nullptr);
    ASSERT_EQ(empty_buffer.device_data<int>(), nullptr);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}