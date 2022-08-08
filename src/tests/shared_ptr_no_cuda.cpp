#include "jutils/shared_ptr.hpp"

#include <gtest/gtest.h>

TEST(NOCUDA_HostSharedPtrTest, toGPUException) {
    auto host_ptr = jutils::make_shared<int>(10);

    bool exception_thrown = false;
    try {
        host_ptr.toGPU();
    } catch(std::exception& exc) {
        exception_thrown = true;
    }
    ASSERT_TRUE(exception_thrown);
}

TEST(NOCUDA_HostSharedPtrTest, fromGPUException) {
    auto host_ptr = jutils::make_shared<int>(10);

    bool exception_thrown = false;
    try {
        host_ptr.fromGPU();
    } catch(std::exception& exc) {
        exception_thrown = true;
    }
    ASSERT_TRUE(exception_thrown);
}


TEST(NOCUDA_UnfiedSharedPtrTest, constructionException) {
    bool exception_thrown = false;
    try {
        auto host_ptr = jutils::make_shared_on_unified<int>(10);
    } catch(std::exception& exc) {
        exception_thrown = true;
    }
    ASSERT_TRUE(exception_thrown);
}

TEST(NOCUDA_SharedPtrTest, countIncrements) {
    auto ptr = jutils::make_shared<int>(10);
    ASSERT_EQ(1, ptr.use_count());
    {
        auto ptr2 = ptr;
        ASSERT_EQ(2, ptr.use_count());
    }
    ASSERT_EQ(1, ptr.use_count());
}

TEST(NOCUDA_SharedPtrTest, nullPtr) {
    jutils::shared_ptr<int> null_ptr;
    ASSERT_TRUE(null_ptr.get() == nullptr);
    ASSERT_TRUE(null_ptr.get_device() == nullptr);
    ASSERT_FALSE(static_cast<bool>(null_ptr));

    null_ptr = jutils::make_shared<int>(20);
    ASSERT_TRUE(null_ptr.get() != nullptr);
    ASSERT_TRUE(null_ptr.get_device() == nullptr);
    ASSERT_TRUE(static_cast<bool>(null_ptr));
}

TEST(NOCUDA_SharedPtrTest, copyAssignment) {
    auto real_ptr = jutils::make_shared<int>(10);
    jutils::shared_ptr<int> null_ptr;

    null_ptr = real_ptr;
    ASSERT_EQ(2, real_ptr.use_count());
    ASSERT_EQ(2, null_ptr.use_count());
    ASSERT_TRUE(null_ptr.get() == real_ptr.get());
}

TEST(NOCUDA_SharedPtrTest, copyConstructor) {
    auto real_ptr = jutils::make_shared<int>(10);
    jutils::shared_ptr<int> real_ptr2(real_ptr);

    ASSERT_EQ(2, real_ptr.use_count());
    ASSERT_EQ(2, real_ptr2.use_count());
    ASSERT_TRUE(real_ptr2.get() == real_ptr.get());
}

TEST(NOCUDA_SharedPtrTest, dereference) {
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

TEST(NOCUDA_SharedPtrTest, moveAssignment) {
    jutils::shared_ptr<int> ptr;
    ptr = jutils::make_shared<int>(20);
    ASSERT_TRUE(ptr.get() != nullptr);
    ASSERT_TRUE(static_cast<bool>(ptr));
}

TEST(NOCUDA_SharedPtrTest, moveConstructor) {
    jutils::shared_ptr<int> ptr(jutils::make_shared<int>(20));
    ASSERT_TRUE(ptr.get() != nullptr);
    ASSERT_TRUE(static_cast<bool>(ptr));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
