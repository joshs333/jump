#include "jutils/multi_array.hpp"
#include "jutils_testing/interopable_test_runner.hpp"

#include <gtest/gtest.h>

TEST(NOCUDA_MultiArray, hostArrayInitializer) {
    jutils::multi_array<int> arr({ {1, 2, 3, 4}, {2,3,4,5}}, jutils::MemType::HOST);
    ASSERT_EQ(1, arr.at(0,0));
    ASSERT_EQ(5, arr.at(1,3));
    ASSERT_EQ(8, arr.size());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(NOCUDA_MultiArray, hostNoInitializer) {
    jutils::multi_array<int> arr(3,4, jutils::MemType::HOST);
    ASSERT_EQ(12, arr.size());
    ASSERT_EQ(3, arr.size_x());
    ASSERT_EQ(4, arr.size_y());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(NOCUDA_MultiArray, hostDefaultInitializer) {
    jutils::multi_array<int> arr(3,4, 77, jutils::MemType::HOST);
    ASSERT_EQ(77, arr.at(0,0));
    ASSERT_EQ(77, arr.at(1,3));
    ASSERT_EQ(77, arr.at(2,3));
    ASSERT_EQ(12, arr.size());
    ASSERT_EQ(3, arr.size_x());
    ASSERT_EQ(4, arr.size_y());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(NOCUDA_MultiArray, deviceArrayInitializer) {
    bool exception = false;
    try {
        jutils::multi_array<int> arr({{1,2,6}, {3,4,5}, {6,7,8}}, jutils::MemType::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_MultiArray, deviceNoInitializer) {
    bool exception = false;
    try {
        jutils::multi_array<int> arr(10,10, jutils::MemType::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_MultiArray, deviceDefaultInitializer) {
    bool exception = false;
    try {
        jutils::multi_array<int> arr(10,10,10,jutils::MemType::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}


TEST(NOCUDA_MultiArray, unfiedArrayInitializer) {
    bool exception = false;
    try {
        jutils::multi_array<int> arr({{1,2,6}, {3,4,5}, {6,7,8}}, jutils::MemType::UNIFIED);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_MultiArray, unfiedNoInitializer) {
    bool exception = false;
    try {
        jutils::multi_array<int> arr(10,10, jutils::MemType::UNIFIED);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_MultiArray, unifiedDefaultInitializer) {
    bool exception = false;
    try {
        jutils::multi_array<int> arr(10,10,10,jutils::MemType::UNIFIED);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}


TEST(NOCUDA_MultiArray, hostSizeCheck) {
    jutils::multi_array<int> arr1(3,4,jutils::MemType::HOST);

    ASSERT_EQ(12, arr1.size());
    ASSERT_EQ(3, arr1.size_x());
    ASSERT_EQ(4, arr1.size_y());
}

TEST(NOCUDA_MultiArray, outOfBoundsTest) {
    jutils::multi_array<int> arr(3,4, 77, jutils::MemType::HOST);
    bool exception = false;
    try {
        arr.at(4,5);
    } catch(std::exception& p) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

// These may not appear to tell much.. but run with cuda-memcheck can be informative
TEST(NOCUDA_MultiArray, hostMoveConstruct) {
    jutils::multi_array<int> arr(jutils::multi_array<int>({{1, 2, 3, 4}}, jutils::MemType::HOST));

    ASSERT_EQ(4, arr.size());
    ASSERT_EQ(1, arr.size_x());
    ASSERT_EQ(4, arr.size_y());
}

TEST(NOCUDA_MultiArray, hostCopyConstruct) {
    jutils::multi_array<int> arr1({{1, 2, 3, 4}}, jutils::MemType::HOST);
    jutils::multi_array<int> arr2(arr1);

    ASSERT_EQ(4, arr2.size());
    ASSERT_EQ(1, arr2.size_x());
    ASSERT_EQ(4, arr2.size_y());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}