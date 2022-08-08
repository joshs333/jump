#include "jutils/multi_array.hpp"
#include "jutils_testing/interopable_test_runner.hpp"

#include <gtest/gtest.h>

namespace multi_array_test {

struct SizeCheck {
    using TestResult = std::size_t;
    using Arguments = jutils::multi_array<int>;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size();
    }
};

struct SizeCheckX {
    using TestResult = std::size_t;
    using Arguments = jutils::multi_array<int>;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size_x();
    }
};

struct SizeCheckY {
    using TestResult = std::size_t;
    using Arguments = jutils::multi_array<int>;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.size_y();
    }
};

struct MemberValueCheck {
    struct Args {
        jutils::multi_array<int> arr;
        std::size_t idx_x;
        std::size_t idx_y;
    };

    using TestResult = int;
    using Arguments = Args;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.arr.at(args.idx_x, args.idx_y);
    }
};

struct MemberValueCheckIndex {
    struct Args {
        jutils::multi_array<int> arr;
        std::size_t idx_x;
        std::size_t idx_y;
    };

    using TestResult = int;
    using Arguments = Args;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.arr[args.idx_x][args.idx_y];
    }
};


} /* namespace multi_array_test */

TEST(MultiArray, hostArrayInitializer) {
    jutils::multi_array<int> arr({ {1, 2, 3, 4}, {2,3,4,5}}, jutils::MemType::HOST);
    ASSERT_EQ(1, arr.at(0,0));
    ASSERT_EQ(1, arr[0][0]);
    ASSERT_EQ(5, arr[1][3]);
    ASSERT_EQ(5, arr.at(1,3));
    ASSERT_EQ(8, arr.size());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(MultiArray, hostNoInitializer) {
    jutils::multi_array<int> arr(3,4, jutils::MemType::HOST);
    ASSERT_EQ(12, arr.size());
    ASSERT_EQ(3, arr.size_x());
    ASSERT_EQ(4, arr.size_y());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(MultiArray, hostDefaultInitializer) {
    jutils::multi_array<int> arr(3,4, 77, jutils::MemType::HOST);
    ASSERT_EQ(77, arr.at(0,0));
    ASSERT_EQ(77, arr.at(1,3));
    ASSERT_EQ(77, arr.at(2,3));
    ASSERT_EQ(12, arr.size());
    ASSERT_EQ(3, arr.size_x());
    ASSERT_EQ(4, arr.size_y());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(MultiArray, deviceArrayInitializer) {
    jutils_testing::InteropableTestRunner<multi_array_test::MemberValueCheck> test;
    multi_array_test::MemberValueCheck::Args args;
    args.arr = jutils::multi_array<int>({{1,2,6}, {3,4,5}, {6,7,8}}, jutils::MemType::DEVICE);
    args.idx_x = 0;
    args.idx_y = 0;

    ASSERT_EQ(1, test.device(args));
    ASSERT_EQ(1, test.device_copy(args));

    args.idx_x = 1;
    args.idx_y = 1;
    ASSERT_EQ(4, test.device(args));
    ASSERT_EQ(4, test.device_copy(args));
}

TEST(MultiArray, deviceArrayInitializerIndex) {
    jutils_testing::InteropableTestRunner<multi_array_test::MemberValueCheckIndex> test;
    multi_array_test::MemberValueCheckIndex::Args args;
    args.arr = jutils::multi_array<int>({{1,2,6}, {3,4,5}, {6,7,8}}, jutils::MemType::DEVICE);
    args.idx_x = 0;
    args.idx_y = 0;

    ASSERT_EQ(1, test.device(args));
    ASSERT_EQ(1, test.device_copy(args));

    args.idx_x = 1;
    args.idx_y = 1;
    ASSERT_EQ(4, test.device(args));
    ASSERT_EQ(4, test.device_copy(args));
}

TEST(MultiArray, deviceNoInitializer) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test;
    jutils::multi_array<int> arr(3,4, jutils::MemType::DEVICE);

    ASSERT_EQ(12, test.host(arr));
    ASSERT_EQ(12, test.device(arr));
    ASSERT_EQ(12, test.device_copy(arr));
}

TEST(MultiArray, deviceDefaultInitializer) {
    jutils_testing::InteropableTestRunner<multi_array_test::MemberValueCheck> test;
    multi_array_test::MemberValueCheck::Args args;
    args.arr = jutils::multi_array<int>(3,4,9,jutils::MemType::DEVICE);
    args.idx_x = 0;
    args.idx_y = 0;

    ASSERT_EQ(9, test.device(args));
    ASSERT_EQ(9, test.device_copy(args));

    args.idx_x = 1;
    args.idx_y = 1;
    ASSERT_EQ(9, test.device(args));
    ASSERT_EQ(9, test.device_copy(args));
}


TEST(MultiArray, unfiedArrayInitializer) {
    jutils_testing::InteropableTestRunner<multi_array_test::MemberValueCheck> test;
    multi_array_test::MemberValueCheck::Args args;
    args.arr = jutils::multi_array<int>({{1,2,6}, {3,4,5}, {6,7,8}}, jutils::MemType::UNIFIED);
    args.idx_x = 0;
    args.idx_y = 0;

    ASSERT_EQ(1, test.host(args));
    ASSERT_EQ(1, test.device(args));
    ASSERT_EQ(1, test.device_copy(args));

    args.idx_x = 1;
    args.idx_y = 1;
    ASSERT_EQ(4, test.host(args));
    ASSERT_EQ(4, test.device(args));
    ASSERT_EQ(4, test.device_copy(args));
}

TEST(MultiArray, unfiedNoInitializer) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test;
    jutils::multi_array<int> arr(3,4, jutils::MemType::UNIFIED);

    ASSERT_EQ(12, test.host(arr));
    ASSERT_EQ(12, test.device(arr));
    ASSERT_EQ(12, test.device_copy(arr));
}

TEST(MultiArray, unifiedDefaultInitializer) {
    jutils_testing::InteropableTestRunner<multi_array_test::MemberValueCheck> test;
    multi_array_test::MemberValueCheck::Args args;
    args.arr = jutils::multi_array<int>(3,4,9,jutils::MemType::UNIFIED);
    args.idx_x = 0;
    args.idx_y = 0;

    ASSERT_EQ(9, test.host(args));
    ASSERT_EQ(9, test.device(args));
    ASSERT_EQ(9, test.device_copy(args));

    args.idx_x = 1;
    args.idx_y = 1;
    ASSERT_EQ(9, test.host(args));
    ASSERT_EQ(9, test.device(args));
    ASSERT_EQ(9, test.device_copy(args));
}


TEST(MultiArray, deviceSizeCheck) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test1;
    jutils::multi_array<int> arr1(3,4, jutils::MemType::DEVICE);

    ASSERT_EQ(12, test1.host(arr1));
    ASSERT_EQ(12, test1.device(arr1));
    ASSERT_EQ(12, test1.device_copy(arr1));

    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheckX> test2;
    jutils::multi_array<int> arr2(3,4, jutils::MemType::DEVICE);

    ASSERT_EQ(3, test2.host(arr2));
    ASSERT_EQ(3, test2.device(arr2));
    ASSERT_EQ(3, test2.device_copy(arr2));

    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheckY> test3;
    jutils::multi_array<int> arr3(3,4, jutils::MemType::DEVICE);

    ASSERT_EQ(4, test3.host(arr3));
    ASSERT_EQ(4, test3.device(arr3));
    ASSERT_EQ(4, test3.device_copy(arr3));
}

TEST(MultiArray, hostSizeCheck) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test1;
    jutils::multi_array<int> arr1(3,4, jutils::MemType::HOST);

    ASSERT_EQ(12, test1.host(arr1));
    ASSERT_EQ(12, test1.device(arr1));
    ASSERT_EQ(12, test1.device_copy(arr1));

    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheckX> test2;
    jutils::multi_array<int> arr2(3,4, jutils::MemType::HOST);

    ASSERT_EQ(3, test2.host(arr2));
    ASSERT_EQ(3, test2.device(arr2));
    ASSERT_EQ(3, test2.device_copy(arr2));

    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheckY> test3;
    jutils::multi_array<int> arr3(3,4, jutils::MemType::HOST);

    ASSERT_EQ(4, test3.host(arr3));
    ASSERT_EQ(4, test3.device(arr3));
    ASSERT_EQ(4, test3.device_copy(arr3));
}

TEST(MultiArray, unfiedSizeCheck) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test1;
    jutils::multi_array<int> arr1(3,4, jutils::MemType::UNIFIED);

    ASSERT_EQ(12, test1.host(arr1));
    ASSERT_EQ(12, test1.device(arr1));
    ASSERT_EQ(12, test1.device_copy(arr1));

    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheckX> test2;
    jutils::multi_array<int> arr2(3,4, jutils::MemType::UNIFIED);

    ASSERT_EQ(3, test2.host(arr2));
    ASSERT_EQ(3, test2.device(arr2));
    ASSERT_EQ(3, test2.device_copy(arr2));

    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheckY> test3;
    jutils::multi_array<int> arr3(3,4, jutils::MemType::UNIFIED);

    ASSERT_EQ(4, test3.host(arr3));
    ASSERT_EQ(4, test3.device(arr3));
    ASSERT_EQ(4, test3.device_copy(arr3));
}


TEST(MultiArray, outOfBoundsTest) {
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
TEST(MultiArray, hostMoveConstruct) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test;
    jutils::multi_array<int> arr(jutils::multi_array<int>({{1, 2, 3, 4}}, jutils::MemType::HOST));

    ASSERT_EQ(4, test.host(arr));
    ASSERT_EQ(4, test.device_copy(arr));
    ASSERT_EQ(4, test.device(arr));
}

TEST(MultiArray, deviceMoveConstruct) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test;
    jutils::multi_array<int> arr(jutils::multi_array<int>({{1, 2, 3, 4, 5}}, jutils::MemType::DEVICE));

    ASSERT_EQ(5, test.host(arr));
    ASSERT_EQ(5, test.device_copy(arr));
    ASSERT_EQ(5, test.device(arr));
}

TEST(MultiArray, unifiedMoveConstruct) {
    jutils_testing::InteropableTestRunner<multi_array_test::SizeCheck> test;
    jutils::multi_array<int> arr(jutils::multi_array<int>({{1, 2, 3, 4, 5, 6}}, jutils::MemType::UNIFIED));

    ASSERT_EQ(6, test.host(arr));
    ASSERT_EQ(6, test.device_copy(arr));
    ASSERT_EQ(6, test.device(arr));
}

TEST(MultiArray, hostCopyConstruct) {
    jutils::multi_array<int> arr1({{1, 2, 3, 4}}, jutils::MemType::HOST);
    jutils::multi_array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data() == arr2.data());
    ASSERT_TRUE(arr1.data_device() == nullptr);
    ASSERT_TRUE(arr2.data_device() == nullptr);
}

TEST(MultiArray, unifiedCopyConstruct) {
    jutils::multi_array<int> arr1({{1, 2, 3, 4}}, jutils::MemType::UNIFIED);
    jutils::multi_array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data() == arr2.data());
    ASSERT_FALSE(arr1.data_device() == arr2.data_device());
    ASSERT_FALSE(arr1.data() == arr2.data_device());
}

TEST(MultiArray, deviceCopyConstruct) {
    jutils::multi_array<int> arr1({{1, 2, 3, 4}}, jutils::MemType::DEVICE);
    jutils::multi_array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data_device() == arr2.data_device());
    ASSERT_TRUE(arr1.data() == nullptr);
    ASSERT_TRUE(arr2.data() == nullptr);
}

TEST(MultiArraySptr, Ptr) {
    jutils::multi_array<int>::ptr({{1, 2, 3, 4}, {3,4,5,6}});
    jutils::multi_array<int>::ptr(10, 100);
    jutils::multi_array<int>::ptr(10, 100, 2);
}

TEST(MultiArraySptr, UnifiedPtr) {
    jutils::multi_array<int>::unified_ptr({{1, 2, 3, 4}, {3,4,5,6}});
    jutils::multi_array<int>::unified_ptr(10, 100);
    jutils::multi_array<int>::unified_ptr(10, 100, 2);
}

TEST(MultiArraySptr, DevicePtr) {
    jutils::multi_array<int>::device_ptr({{1, 2, 3, 4}, {3,4,5,6}});
    jutils::multi_array<int>::device_ptr(10, 100);
    jutils::multi_array<int>::device_ptr(10, 100, 2);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}