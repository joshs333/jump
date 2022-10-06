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

struct SptrMemberValueCheck {
    struct Args {
        jutils::array<int>::sptr arr;
        std::size_t idx;
    };

    using TestResult = int;
    using Arguments = Args;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.arr->at(args.idx);
    }
};


} /* namespace array_test */

TEST(Array, hostArrayInitializer) {
    jutils::array<int> arr({1, 2, 3, 4}, jutils::MemType::HOST);
    ASSERT_EQ(1, arr[0]);
    ASSERT_EQ(4, arr[3]);
    ASSERT_EQ(1, arr.at(0));
    ASSERT_EQ(4, arr.at(3));
    ASSERT_EQ(4, arr.size());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(Array, hostNoInitializer) {
    jutils::array<int> arr(5);
    ASSERT_EQ(5, arr.size());
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(Array, hostDefaultInitializer) {
    jutils::array<int> arr(7, 23);
    ASSERT_EQ(7, arr.size());
    ASSERT_EQ(23, arr[0]);
    ASSERT_EQ(23, arr[3]);
    ASSERT_EQ(23, arr.at(0));
    ASSERT_EQ(23, arr.at(3));
    ASSERT_EQ(arr.data_device() , nullptr);
}

TEST(Array, unifiedArrayInitializer) {
    array_test::MemberValueCheck::Args args;
    args.arr = jutils::array<int>({1, 2, 3, 4}, jutils::MemType::UNIFIED);

    jutils_testing::InteropableTestRunner<array_test::MemberValueCheck> test;
    args.idx = 0;
    ASSERT_EQ(1, test.host(args));
    ASSERT_EQ(1, test.device(args));
    args.idx = 3;
    ASSERT_EQ(4, test.host(args));
    ASSERT_EQ(4, test.device_copy(args));
    ASSERT_EQ(args.arr.data() , args.arr.data_device());
}

TEST(Array, unifiedDefaultInitializer) {
    array_test::MemberValueCheck::Args args;
    args.arr = jutils::array<int>(7, 49, jutils::MemType::UNIFIED);

    jutils_testing::InteropableTestRunner<array_test::MemberValueCheck> test;
    args.idx = 0;
    ASSERT_EQ(49, test.host(args));
    ASSERT_EQ(49, test.device(args));
    ASSERT_EQ(49, test.device_copy(args));
    args.idx = 3;
    ASSERT_EQ(49, test.host(args));
    ASSERT_EQ(49, test.device(args));
    ASSERT_EQ(49, test.device_copy(args));
    ASSERT_EQ(args.arr.data() , args.arr.data_device());
}

TEST(Array, hostArraySizeCheck) {
    jutils_testing::InteropableTestRunner<array_test::SizeCheck> test;
    jutils::array<int> arr({1, 2, 3, 4});

    ASSERT_EQ(4, test.host(arr));
    ASSERT_EQ(4, test.device_copy(arr));
    ASSERT_EQ(4, test.device(arr));
}

TEST(Array, unfiedNoInitializerArraySizeCheck) {
    jutils_testing::InteropableTestRunner<array_test::SizeCheck> test;
    jutils::array<int> arr(4, jutils::MemType::UNIFIED);

    ASSERT_EQ(4, test.host(arr));
    ASSERT_EQ(4, test.device_copy(arr));
    ASSERT_EQ(4, test.device(arr));
}

// These may not appear to tell much.. but run with cuda-memcheck can be informative
TEST(Array, hostMoveConstruct) {
    jutils_testing::InteropableTestRunner<array_test::SizeCheck> test;
    jutils::array<int> arr(jutils::array<int>({1, 2, 3, 4}, jutils::MemType::HOST));

    ASSERT_EQ(4, test.host(arr));
    ASSERT_EQ(4, test.device_copy(arr));
    ASSERT_EQ(4, test.device(arr));
}

TEST(Array, deviceMoveConstruct) {
    jutils_testing::InteropableTestRunner<array_test::SizeCheck> test;
    jutils::array<int> arr(jutils::array<int>({1, 2, 3, 4}, jutils::MemType::DEVICE));

    ASSERT_EQ(4, test.host(arr));
    ASSERT_EQ(4, test.device_copy(arr));
    ASSERT_EQ(4, test.device(arr));
}

TEST(Array, unifiedMoveConstruct) {
    jutils_testing::InteropableTestRunner<array_test::SizeCheck> test;
    jutils::array<int> arr(jutils::array<int>({1, 2, 3, 4}, jutils::MemType::UNIFIED));

    ASSERT_EQ(4, test.host(arr));
    ASSERT_EQ(4, test.device_copy(arr));
    ASSERT_EQ(4, test.device(arr));
}


TEST(Array, hostCopyConstruct) {
    jutils::array<int> arr1({1, 2, 3, 4}, jutils::MemType::HOST);
    jutils::array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data() == arr2.data());
    ASSERT_TRUE(arr1.data_device() == nullptr);
    ASSERT_TRUE(arr2.data_device() == nullptr);
}

TEST(Array, unifiedCopyConstruct) {
    jutils::array<int> arr1({1, 2, 3, 4}, jutils::MemType::UNIFIED);
    jutils::array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data() == arr2.data());
    ASSERT_FALSE(arr1.data_device() == arr2.data_device());
    ASSERT_FALSE(arr1.data() == arr2.data_device());
}


TEST(Array, deviceCopyConstruct) {
    jutils::array<int> arr1({1, 2, 3, 4}, jutils::MemType::DEVICE);
    jutils::array<int> arr2(arr1);

    ASSERT_EQ(arr1.size(), arr2.size());
    ASSERT_FALSE(arr1.data_device() == arr2.data_device());
    ASSERT_TRUE(arr1.data() == nullptr);
    ASSERT_TRUE(arr2.data() == nullptr);
}

TEST(Array, deviceFromGPU) {
    array_test::MemberValueCheck::Args args;
    args.arr = jutils::array<int>({1, 2, 3, 4}, jutils::MemType::DEVICE);

    ASSERT_TRUE(args.arr.data_device() != nullptr);
    ASSERT_TRUE(args.arr.data() == nullptr);
    args.arr.fromGPU();
    ASSERT_TRUE(args.arr.data() != nullptr);

    jutils_testing::InteropableTestRunner<array_test::MemberValueCheck> test;
    args.idx = 0;
    ASSERT_EQ(1, test.host(args));
    ASSERT_EQ(1, test.device(args));
    ASSERT_EQ(1, test.device_copy(args));
    args.idx = 3;
    ASSERT_EQ(4, test.host(args));
    ASSERT_EQ(4, test.device(args));
    ASSERT_EQ(4, test.device_copy(args));
}

TEST(Array, hostToGPU) {
    array_test::MemberValueCheck::Args args;
    args.arr = jutils::array<int>({1, 2, 3, 4}, jutils::MemType::HOST);

    ASSERT_TRUE(args.arr.data() != nullptr);
    ASSERT_TRUE(args.arr.data_device() == nullptr);
    args.arr.toGPU();
    ASSERT_TRUE(args.arr.data_device() != nullptr);

    jutils_testing::InteropableTestRunner<array_test::MemberValueCheck> test;
    args.idx = 0;
    ASSERT_EQ(1, test.host(args));
    ASSERT_EQ(1, test.device(args));
    ASSERT_EQ(1, test.device_copy(args));
    args.idx = 3;
    ASSERT_EQ(4, test.host(args));
    ASSERT_EQ(4, test.device(args));
    ASSERT_EQ(4, test.device_copy(args));
}


TEST(ArraySptr, Ptr) {
    jutils::array<int>::ptr({1, 2, 3, 4});
    jutils::array<int>::ptr(10);
    jutils::array<int>::ptr(10, 11);
}

TEST(ArraySptr, UnifiedPtr) {
    jutils::array<int>::unified_ptr({1, 2, 3, 4});
    jutils::array<int>::unified_ptr(10);
    jutils::array<int>::unified_ptr(10, 11);
}

TEST(ArraySptr, DevicePtr) {
    jutils::array<int>::device_ptr({1, 2, 3, 4});
    jutils::array<int>::device_ptr(10);
    jutils::array<int>::device_ptr(10, 11);
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}