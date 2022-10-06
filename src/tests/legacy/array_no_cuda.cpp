#include "jutils/array.hpp"
#include "jutils_testing/interopable_test_runner.hpp"

#include <gtest/gtest.h>

namespace array_test {

struct SizeCheck {
    using TestResult = std::size_t;
    using Arguments = jutils::array<int>;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size();
    }
};

struct MemberValueCheck {
    struct Args {
        jutils::array<int> arr;
        std::size_t idx;
    };

    using TestResult = int;
    using Arguments = Args;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.arr[args.idx];
    }
};

} /* namespace array_test */

TEST(NOCUDA_SharedArray, hostArrayInitializer) {
    jutils::array<int> arr({1, 2, 3, 4}, jutils::MemType::HOST);
    ASSERT_EQ(1, arr[0]);
    ASSERT_EQ(4, arr[3]);
    ASSERT_EQ(1, arr.at(0));
    ASSERT_EQ(4, arr.at(3));
    ASSERT_EQ(4, arr.size());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(NOCUDA_SharedArray, hostNoInitializer) {
    jutils::array<int> arr(5);
    ASSERT_EQ(5, arr.size());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(NOCUDA_SharedArray, hostDefaultInitializer) {
    jutils::array<int> arr(7, 23);
    ASSERT_EQ(7, arr.size());
    ASSERT_EQ(23, arr[0]);
    ASSERT_EQ(23, arr[3]);
    ASSERT_EQ(23, arr.at(0));
    ASSERT_EQ(23, arr.at(3));
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(NOCUDA_SharedArray, unifiedArrayInitializer) {
    bool exception = false;
    try {
        jutils::array<int> arr({1, 2, 3, 4}, jutils::MemType::UNIFIED);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_SharedArray, unifiedNoInitializer) {
    bool exception = false;
    try {
        jutils::array<int> arr(10, jutils::MemType::UNIFIED);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_SharedArray, unifiedDefaultInitializer) {
    bool exception = false;
    try {
        jutils::array<int> arr(10, 49, jutils::MemType::UNIFIED);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_SharedArray, deviceArrayInitializer) {
    bool exception = false;
    try {
        jutils::array<int> arr({1, 2, 3, 4}, jutils::MemType::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_SharedArray, deviceNoInitializer) {
    bool exception = false;
    try {
        jutils::array<int> arr(10, jutils::MemType::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(NOCUDA_SharedArray, deviceDefaultInitializer) {
    bool exception = false;
    try {
        jutils::array<int> arr(10, 49, jutils::MemType::DEVICE);
    } catch(std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

// These may not appear to tell much.. but run with cuda-memcheck can be informative
TEST(NOCUDA_SharedArray, hostMoveConstruct) {
    jutils_testing::InteropableTestRunner<array_test::SizeCheck> test;
    jutils::array<int> arr(jutils::array<int>({1, 2, 3, 4}, jutils::MemType::HOST));

    ASSERT_EQ(4, test.host(arr));
}

TEST(NOCUDA_SharedArray, hostCopyConstruct) {
    jutils::array<int> arr1({1, 2, 3, 4}, jutils::MemType::HOST);
    jutils::array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data() == arr2.data());
    ASSERT_TRUE(arr1.data_device() == nullptr);
    ASSERT_TRUE(arr2.data_device() == nullptr);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}