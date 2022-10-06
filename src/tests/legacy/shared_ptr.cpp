#include "jutils/shared_ptr.hpp"
#include "jutils_testing/interopable_test_runner.hpp"

#include <gtest/gtest.h>

namespace shared_ptr_test {

class InteropableClass {
public:
    InteropableClass(jutils::shared_ptr<int> value):
        value_(value)
    {}

    GPU_COMPATIBLE
    int getValue() const {
        return (*value_)++;
    }

    bool toGPU(const InteropableClass* ptr) {
        value_.toGPU();
        return true;
    }

    bool fromGPU(const InteropableClass* ptr) {
        value_.fromGPU();
        // The only data we actually have is in the value_ shared_ptr
        // so nothing else needs copied
        return false;
    }

private:
    jutils::shared_ptr<int> value_;

};

struct SimpleInteropableTest {
    using TestResult = int;
    using Arguments = jutils::shared_ptr<int>;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = (*args)++;
    }
};

struct ComplexInteropableTest {
    using TestResult = int;
    using Arguments = InteropableClass;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args.getValue();
    }
};

struct ComplexInteropableTestPtr {
    using TestResult = int;
    using Arguments = jutils::shared_ptr<InteropableClass>;

    GPU_COMPATIBLE
    static void test(const Arguments& args, TestResult& result) {
        result = args->getValue();
    }
};


} /* namespace shared_ptr_test */

TEST(HostSharedPtrTest, SimpleInteropability) {
    auto host_ptr = jutils::make_shared<int>(10);

    jutils_testing::InteropableTestRunner<shared_ptr_test::SimpleInteropableTest> test;
    ASSERT_EQ(10, test.host(host_ptr));
    host_ptr.toGPU();
    ASSERT_EQ(11, test.device(host_ptr));
    ASSERT_EQ(11, test.host(host_ptr));
    ASSERT_EQ(12, test.host(host_ptr));
    host_ptr.fromGPU();
    ASSERT_EQ(12, test.host(host_ptr));
}

TEST(HostSharedPtrTest, ComplexInteropability) {
    auto underlaying_ptr = jutils::make_shared<int>(10);
    shared_ptr_test::InteropableClass object(underlaying_ptr);

    jutils_testing::InteropableTestRunner<shared_ptr_test::ComplexInteropableTest> test;
    ASSERT_EQ(10, test.host(object));
    object.toGPU(nullptr);
    ASSERT_EQ(11, test.device(object));
    ASSERT_EQ(11, test.host(object));
    ASSERT_EQ(12, test.host(object));
    object.fromGPU(nullptr);
    ASSERT_EQ(12, test.host(object));
}

TEST(HostSharedPtrTest, ComplexInteropabilityPtr) {
    auto underlaying_ptr = jutils::make_shared<int>(10);
    auto object = jutils::make_shared<shared_ptr_test::InteropableClass>(underlaying_ptr);

    jutils_testing::InteropableTestRunner<shared_ptr_test::ComplexInteropableTestPtr> test;
    ASSERT_EQ(10, test.host(object));
    object.toGPU();
    ASSERT_EQ(11, test.device(object));
    ASSERT_EQ(12, test.device_copy(object));
    ASSERT_EQ(11, test.host(object));
    ASSERT_EQ(12, test.host(object));
    object.fromGPU();
    ASSERT_EQ(13, test.host(object));
}

TEST(UnifiedSharedPtrTest, SimpleInteropability) {
    auto host_ptr = jutils::make_shared_on_unified<int>(10);

    jutils_testing::InteropableTestRunner<shared_ptr_test::SimpleInteropableTest> test;
    ASSERT_EQ(10, test.host(host_ptr));
    ASSERT_EQ(11, test.device(host_ptr));
    ASSERT_EQ(12, test.host(host_ptr));
}

TEST(UnifiedSharedPtrTest, ComplexInteropability) {
    auto underlaying_ptr = jutils::make_shared_on_unified<int>(10);
    shared_ptr_test::InteropableClass object(underlaying_ptr);

    jutils_testing::InteropableTestRunner<shared_ptr_test::ComplexInteropableTest> test;
    ASSERT_EQ(10, test.host(object));
    ASSERT_EQ(11, test.device(object));
    ASSERT_EQ(12, test.host(object));
}

TEST(UnifiedSharedPtrTest, ComplexInteropabilityPtr) {
    auto underlaying_ptr = jutils::make_shared_on_unified<int>(10);
    auto object = jutils::make_shared_on_unified<shared_ptr_test::InteropableClass>(underlaying_ptr);

    jutils_testing::InteropableTestRunner<shared_ptr_test::ComplexInteropableTestPtr> test;
    ASSERT_EQ(10, test.host(object));
    ASSERT_EQ(11, test.device(object));
    ASSERT_EQ(12, test.host(object));
}

TEST(UnifiedSharedPtrTest, HostUnderlayingA) {
    auto underlaying_ptr = jutils::make_shared<int>(10);
    auto object = jutils::make_shared_on_unified<shared_ptr_test::InteropableClass>(underlaying_ptr);

    jutils_testing::InteropableTestRunner<shared_ptr_test::ComplexInteropableTestPtr> test;
    ASSERT_EQ(10, test.host(object));
    object.toGPU();
    ASSERT_EQ(11, test.device(object));
    object.fromGPU();
    ASSERT_EQ(12, test.host(object));
}

TEST(UnifiedSharedPtrTest, HostUnderlayingB) {
    auto underlaying_ptr = jutils::make_shared<int>(10);
    auto object = jutils::make_shared_on_unified<shared_ptr_test::InteropableClass>(underlaying_ptr);

    jutils_testing::InteropableTestRunner<shared_ptr_test::ComplexInteropableTestPtr> test;
    ASSERT_EQ(10, test.host(object));
    object->toGPU(nullptr);
    ASSERT_EQ(11, test.device(object));
    object->fromGPU(nullptr);
    ASSERT_EQ(12, test.host(object));
}

TEST(SharedPtrTest, countIncrements) {
    auto ptr = jutils::make_shared<int>(10);
    ASSERT_EQ(1, ptr.use_count());
    {
        auto ptr2 = ptr;
        ASSERT_EQ(2, ptr.use_count());
    }
    ASSERT_EQ(1, ptr.use_count());
}

TEST(SharedPtrTest, nullPtr) {
    jutils::shared_ptr<int> null_ptr;
    ASSERT_TRUE(null_ptr.get() == nullptr);
    ASSERT_TRUE(null_ptr.get_device() == nullptr);
    ASSERT_FALSE(static_cast<bool>(null_ptr));

    null_ptr = jutils::make_shared<int>(20);
    ASSERT_TRUE(null_ptr.get() != nullptr);
    ASSERT_TRUE(null_ptr.get_device() == nullptr);
    ASSERT_TRUE(static_cast<bool>(null_ptr));
}

TEST(SharedPtrTest, copyAssignment) {
    auto real_ptr = jutils::make_shared<int>(10);
    jutils::shared_ptr<int> null_ptr;

    null_ptr = real_ptr;
    ASSERT_EQ(2, real_ptr.use_count());
    ASSERT_EQ(2, null_ptr.use_count());
    ASSERT_TRUE(null_ptr.get() == real_ptr.get());
}

TEST(SharedPtrTest, copyConstructor) {
    auto real_ptr = jutils::make_shared<int>(10);
    jutils::shared_ptr<int> real_ptr2(real_ptr);

    ASSERT_EQ(2, real_ptr.use_count());
    ASSERT_EQ(2, real_ptr2.use_count());
    ASSERT_TRUE(real_ptr2.get() == real_ptr.get());
}

TEST(SharedPtrTest, dereference) {
    auto ptr = jutils::make_shared<int>(10);
    ASSERT_EQ(10, *ptr);

    auto ptr2 = ptr;
    *ptr2 = 20;
    ASSERT_EQ(20, *ptr);

    auto ptr3 = ptr;
    *(ptr3.get()) = 30;
    ASSERT_EQ(30, *ptr);
    ASSERT_EQ(30, *ptr2);
}

TEST(SharedPtrTest, moveAssignment) {
    jutils::shared_ptr<int> ptr;
    ptr = jutils::make_shared<int>(20);
    ASSERT_TRUE(ptr.get() != nullptr);
    ASSERT_TRUE(static_cast<bool>(ptr));
}

TEST(SharedPtrTest, moveConstructor) {
    jutils::shared_ptr<int> ptr(jutils::make_shared<int>(20));
    ASSERT_TRUE(ptr.get() != nullptr);
    ASSERT_TRUE(static_cast<bool>(ptr));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
