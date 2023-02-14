/**
 * @file memory_buffer.cpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief tests for the jump::memory_buffer class
 * @date 2022-10-06
 */
#include <jump/multi_array.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME MultiArray
#else
    #define TEST_SUITE_NAME MultiArrayNoCuda
#endif


TEST(TEST_SUITE_NAME, constructInitializer) {
    jump::multi_array<int> ar({10, 10, 10});

    ASSERT_EQ(ar.dims(), 3);
    for(int i = 0; i < ar.dims(); ++i) {
        ASSERT_EQ(ar.shape(i), 10);
    }
}

TEST(TEST_SUITE_NAME, exceedMaxDims) {
    bool exception = false;
    try {
        jump::multi_array<int, 2> ar1({1, 2, 3, 4});
    } catch (std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);

    exception = false;
    try {
        jump::multi_array<int, 4> ar1({1, 2, 3, 4});
    } catch (std::exception& e) {
        exception = true;
    }
    ASSERT_FALSE(exception);
}

TEST(TEST_SUITE_NAME, shapeExceedRange) {
    jump::multi_array<int> ar({10, 10, 10});

    bool exception = false;
    try {
        ar.shape(3);
    } catch (std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, exceedIndexRange) {
    jump::multi_array<int> ar({10, 10, 10});

    ar.at(9, 9, 9);
    bool exception = false;
    try {
        ar.at(10, 10, 10);
    } catch (std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
}

TEST(TEST_SUITE_NAME, defaultConstructorCheckValues) {
    jump::multi_array<int> ar({13, 4, 10}, 10);

    ASSERT_EQ(ar.size(), 13 * 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            for(int k = 0; k < ar.shape(2); ++k) {
                ASSERT_EQ(ar.at(i, j, k), 10);
            }   
        }    
    }
}

TEST(TEST_SUITE_NAME, copyConstructorCheck) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto ar = ar_orig;

    ASSERT_EQ(ar.size(), 13 * 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            for(int k = 0; k < ar.shape(2); ++k) {
                ASSERT_EQ(ar.at(i, j, k), 10);
            }   
        }    
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
