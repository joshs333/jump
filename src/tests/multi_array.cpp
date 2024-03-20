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

TEST(TEST_SUITE_NAME, moveAssignment) {
    jump::multi_array<int> ar({2, 3});

    bool exception = false;
    try {
        ar = jump::multi_array<int>({1, 2});
    } catch(const std::exception& e) {
        exception = true;
    }
    ASSERT_FALSE(exception);
}

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

TEST(TEST_SUITE_NAME, fullArrayView) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto ar = ar_orig.view();

    ASSERT_EQ(ar.size(), 13 * 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            for(int k = 0; k < ar.shape(2); ++k) {
                ASSERT_EQ(ar.at(i, j, k), 10);
            }
        }
    }
}


TEST(TEST_SUITE_NAME, partialArrayView) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto ar = ar_orig.view(10);

    ASSERT_EQ(ar.size(), 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            ASSERT_EQ(ar.at(i, j), 10);
        }
    }
}

TEST(TEST_SUITE_NAME, viewOfViewFull) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto view = ar_orig.view();
    auto ar = view.view();

    ASSERT_EQ(ar.size(), 13 * 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            for(int k = 0; k < ar.shape(2); ++k) {
                ASSERT_EQ(ar.at(i, j, k), 10);
            }
        }
    }
}

TEST(TEST_SUITE_NAME, viewOfViewPartial1) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto view = ar_orig.view(10);
    auto ar = view.view();

    ASSERT_EQ(ar.size(), 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            ASSERT_EQ(ar.at(i, j), 10);
        }
    }
}

TEST(TEST_SUITE_NAME, viewOfViewPartial2) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto view = ar_orig.view();
    auto ar = view.view(10);

    ASSERT_EQ(ar.size(), 4 * 10);
    for(int i = 0; i < ar.shape(0); ++i) {
        for(int j = 0; j < ar.shape(1); ++j) {
            ASSERT_EQ(ar.at(i, j), 10);
        }
    }
}

TEST(TEST_SUITE_NAME, viewOfViewPartial3) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto view = ar_orig.view(10);
    auto ar = view.view(1);

    ASSERT_EQ(ar.size(), 10);
    for(int j = 0; j < ar.shape(0); ++j) {
        ASSERT_EQ(ar.at(j), 10);
    }
}

TEST(TEST_SUITE_NAME, viewValueModification) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    auto view = ar_orig.view(10);
    auto ar = view.view(1);

    ASSERT_EQ(ar.at(0), 10);
    ar.at(0) = 1;
    ar.at(4) = 2;
    bool exception = false;
    try {
        ar.at(10) = 5;
    } catch(const std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE(exception);
    ASSERT_EQ(ar.at(0), 1);
    ASSERT_EQ(view.at(1, 0), 1);
    ASSERT_EQ(ar_orig.at(10, 1, 0), 1);
    ASSERT_EQ(ar.at(4), 2);
    ASSERT_EQ(view.at(1, 4), 2);
    ASSERT_EQ(ar_orig.at(10, 1, 4), 2);
}

TEST(TEST_SUITE_NAME, deviceOperation) {
    bool exception = false;
    try {
        jump::multi_array<int> ar_orig({13, 4, 10}, jump::memory_t::DEVICE);
    } catch(const std::exception& e) {
        exception = true;
    }
    ASSERT_TRUE((!exception && jump::cuda_enabled()) || (exception && !jump::cuda_enabled()));
}

TEST(TEST_SUITE_NAME, commaOperator) {
    jump::multi_array<int> ar_orig({2, 3});
    ar_orig << 1, 2, 3, 4, 5, 6;

    ASSERT_EQ(ar_orig.at(0, 0), 1);
    ASSERT_EQ(ar_orig.at(0, 1), 2);
    ASSERT_EQ(ar_orig.at(0, 2), 3);
    ASSERT_EQ(ar_orig.at(1, 0), 4);
    ASSERT_EQ(ar_orig.at(1, 1), 5);
    ASSERT_EQ(ar_orig.at(1, 2), 6);

    bool ar_exception = false;
    try {
        ar_orig << 1, 2, 3, 4, 5, 6, 7;
    } catch(const std::exception& e) {
        ar_exception = true;
    }
    ASSERT_TRUE(ar_exception);
    
    auto v = ar_orig.view(1);
    v << 8, 9, 10;
    ASSERT_EQ(v[0], 8);
    ASSERT_EQ(v[1], 9);
    ASSERT_EQ(v[2], 10);
    ASSERT_EQ(ar_orig.at(1, 0), 8);
    ASSERT_EQ(ar_orig.at(1, 1), 9);
    ASSERT_EQ(ar_orig.at(1, 2), 10);
    bool view_exception = false;
    try {
        v << 8, 9, 10, 11;
    } catch(const std::exception& e) {
        view_exception = true;
    }
    ASSERT_TRUE(view_exception);
}

bool useView(jump::multi_array_view<int> arr) {
    return true;
}

bool useViewConst(const jump::multi_array_view<int>& arr) {
    return true;
}

// doesn't work
bool useViewRef(jump::multi_array_view<int>& arr) {
    return true;
}

TEST(TEST_SUITE_NAME, implicitViewConversion) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);
    ASSERT_TRUE(useView(ar_orig));
    ASSERT_TRUE(useViewConst(ar_orig));
};


struct complex_struct {
    unsigned int a;
};

TEST(TEST_SUITE_NAME, recast) {
    jump::multi_array<int> ar_orig({13, 4, 10}, 10);

    auto r2 = ar_orig.recast<unsigned int>();
    // auto r3 = ar_orig.recast<long int>(); // just won't compile, which is good!
    // auto r3 = ar_orig.recast<complex_struct>(); // just won't compile, which is good!
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
