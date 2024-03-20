/**
 * @file variant.cpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief tests for the jump::variant class
 * @date 2024-02-06
 */
#include <jump/variant.hpp>
#include <jump/array.hpp>
#include <jump_testing/interopable_test_runner.hpp>

#include <gtest/gtest.h>

#ifdef JUMP_ENABLE_CUDA
    #define TEST_SUITE_NAME Variant
#else
    #define TEST_SUITE_NAME VariantNoCuda
#endif

namespace variant_testing {

struct PerformIndexTest {
    using Arguments = int;
    struct TestResult {
        int a;
        int b;
        int c;
        int d;
    };

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result.a = jump::_variant_helpers::type_index<int, int, double, float>::value;
        result.b = jump::_variant_helpers::type_index<double, int, double, float>::value;
        result.c = jump::_variant_helpers::type_index<float, int, double, float>::value;
        result.d = jump::_variant_helpers::type_index<int8_t, int, double, float>::value;
    }
};

template<typename GetType, typename... Types>
struct VariantGetSet {
    using Arguments = jump::variant<Types...>;
    struct TestResult {
        int index;
        GetType value;
    };

    JUMP_INTEROPABLE
    static void test(const Arguments& args, TestResult& result) {
        result.index = args.index();
        result.value = args.template get<GetType>();
    }
};


} /* namespace variant_testing */


TEST(TEST_SUITE_NAME, typeIndexTest) {
    using TestRunner = jump_testing::InteropableTestRunner<variant_testing::PerformIndexTest>;

    TestRunner runner;
    auto host_val = runner.host(0);

    ASSERT_EQ(host_val.a, 0);
    ASSERT_EQ(host_val.b, 1);
    ASSERT_EQ(host_val.c, 2);
    ASSERT_TRUE(host_val.d < 0);

    if constexpr(jump::cuda_enabled()) {
        auto device_val = runner.device(0);
        ASSERT_EQ(device_val.a, 0);
        ASSERT_EQ(device_val.b, 1);
        ASSERT_EQ(device_val.c, 2);
        ASSERT_TRUE(device_val.d < 0);

        auto device_copy_val = runner.device_copy(0);
        ASSERT_EQ(device_copy_val.a, 0);
        ASSERT_EQ(device_copy_val.b, 1);
        ASSERT_EQ(device_copy_val.c, 2);
        ASSERT_TRUE(device_copy_val.d < 0);
    }
}


TEST(TEST_SUITE_NAME, variantSetTest) {
    using IntTest = variant_testing::VariantGetSet<int, int, double, float>;
    using IntTestRunner = jump_testing::InteropableTestRunner<IntTest>;

    jump::variant<int, double, float> int_var(static_cast<int>(10));
    auto host_result = IntTestRunner().host(int_var);
    ASSERT_EQ(host_result.index, 0);
    ASSERT_EQ(host_result.value, 10);


    using FloatTest = variant_testing::VariantGetSet<float, int, float, double>;
    using FloatTestRunner = jump_testing::InteropableTestRunner<FloatTest>;
    jump::variant<int, float, double> float_var(static_cast<float>(12.2));
    auto float_host_result = FloatTestRunner().host(float_var);
    ASSERT_EQ(float_host_result.index, 1);
    ASSERT_NEAR(float_host_result.value, 12.2, 1e-6);

    if constexpr(jump::cuda_enabled()) {
        host_result = IntTestRunner().device(int_var);
        ASSERT_EQ(host_result.index, 0);
        ASSERT_EQ(host_result.value, 10);
        
        float_host_result = FloatTestRunner().device(float_var);
        ASSERT_EQ(float_host_result.index, 1);
        ASSERT_NEAR(float_host_result.value, 12.2, 1e-6);

        host_result = IntTestRunner().device_copy(int_var);
        ASSERT_EQ(host_result.index, 0);
        ASSERT_EQ(host_result.value, 10);
        
        float_host_result = FloatTestRunner().device_copy(float_var);
        ASSERT_EQ(float_host_result.index, 1);
        ASSERT_NEAR(float_host_result.value, 12.2, 1e-6);
    }
}

TEST(TEST_SUITE_NAME, variantOperations) {
    jump::variant<int, float, double> float_var(1.0);
    float_var = 2.0;


    jump::variant<int, float, double> float_var2;
    float_var2 = float_var;
    float_var = std::move(float_var2);
}


struct generic_visitor {
    bool* val_to_set = nullptr;

    template<typename T>
    void visit(T& value) {
        if(val_to_set)
            *val_to_set = true;
    }
};

TEST(TEST_SUITE_NAME, visitorDefined) {
    struct int_visitor {
        bool* val_to_set = nullptr;

        void visit(int& value) {
            if(val_to_set)
                *val_to_set = true;
        }
    };

    bool v = jump::_variant_helpers::visitor_defined<generic_visitor, int&>();
    ASSERT_TRUE(v);
    v = jump::_variant_helpers::visitor_defined<generic_visitor, float&>();
    ASSERT_TRUE(v);

    v = jump::_variant_helpers::visitor_defined<int_visitor, int&>();
    ASSERT_TRUE(v);
    v = jump::_variant_helpers::visitor_defined<int_visitor, float&>();
    ASSERT_FALSE(v);
}


TEST(TEST_SUITE_NAME, performVisit) {
    struct int_visitor {
        bool* val_to_set = nullptr;

        void visit(int& value) {
            if(val_to_set)
                *val_to_set = true;
        }
    };

    jump::variant<float, int, bool> v_f(0.0f);
    bool v_f_generic_success = false;
    bool v_f_int_success = false;
    v_f.visit(generic_visitor{&v_f_generic_success});
    v_f.visit(int_visitor{&v_f_int_success});
    ASSERT_TRUE(v_f_generic_success);
    ASSERT_FALSE(v_f_int_success);


    jump::variant<float, int, bool> v_i(0);
    bool v_i_generic_success = false;
    bool v_i_int_success = false;
    v_i.visit(generic_visitor{&v_i_generic_success});
    v_i.visit(int_visitor{&v_i_int_success});
    ASSERT_TRUE(v_i_generic_success);
    ASSERT_TRUE(v_i_int_success);
}

class A {
public:
    A() {}
};

class B {
public:
    B() {}
};

TEST(TEST_SUITE_NAME, arrayOfVariants) {
    jump::array<jump::variant<A, B>> arr;
    A a;
    const jump::variant<A, B> v(a);
    arr.push_back(v);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
