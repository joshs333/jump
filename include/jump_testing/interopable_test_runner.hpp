/**
 * @file interopable_test_runner.hpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief provides a small shim that can call and get results
 *  for an interopable test (GPU / CPU)
 * @date 2022-04-29
 */
#ifndef JCONTROLS_TESTING_INTEROPABLE_TESTING_HPP_
#define JCONTROLS_TESTING_INTEROPABLE_TESTING_HPP_

#include "jcontrols/cuda_interface.hpp"
#include <stdexcept>

namespace jcontrols_testing {

//! Internal namespace of helpers for the InteropableTestRunner
namespace interopable_test_runner_helpers {

#if CUDA_AVAILABLE
    //! Should be called with <<<1,1>>>
    template<typename Test>
    __global__ void runTest(
        const typename Test::Arguments args,
        typename Test::TestResult* result_location
    ) {
        Test::test(args, *result_location);
    } /* runTest */

    //! Should be called with <<<1,1>>>
    template<typename Test>
    __global__ void runTestViaCopy(
        typename Test::Arguments* args,
        typename Test::TestResult* result_location
    ) {
        Test::test(*args, *result_location);
    } /* runTest */
#endif

} /* namepsace interopable_test_runner_helpers */


//! The example structure of an interopable test
struct InteropableTest {
    using TestResult = int;
    using Arguments = int;

    GPU_COMPATIBLE
    void test(const Arguments& args, TestResult& result) {
        #if ON_GPU
            result = 10;
        #else
            result = 20;
        #endif        
    }
}; /* InteropableTest */


//! Test runner that abstracts away allocating cuda memory
//! calling a kernel and transferring results away
template<typename Test = InteropableTest>
class InteropableTestRunner {
public:
    //! Get results for the test running on host
    typename Test::TestResult 
    host(const typename Test::Arguments& arguments) {
        typename Test::TestResult result;
        Test::test(arguments, result);
        return result;
    }

    //! Get results for the test running on device, arguments passed directly to kernel
    typename Test::TestResult 
    device(const typename Test::Arguments& arguments) {
        #if CUDA_AVAILABLE
            typename Test::TestResult* device_result_ptr;
            typename Test::TestResult result;

            cudaMalloc(&device_result_ptr, sizeof(typename Test::TestResult));
            interopable_test_runner_helpers::runTest<Test><<<1,1>>>(arguments, device_result_ptr);
            cudaDeviceSynchronize();

            cudaMemcpy(&result, device_result_ptr, sizeof(typename Test::TestResult), cudaMemcpyDeviceToHost);
            cudaFree(device_result_ptr);
            
            return result;
        #else
            throw std::runtime_error("Cuda is not available to run device test");
        #endif
    }

    //! Get results for the test running on device, arguments mem-copied to gpu and ptr passed to kernel
    typename Test::TestResult 
    device_copy(const typename Test::Arguments& arguments) {
        #if CUDA_AVAILABLE
            typename Test::TestResult* device_result_ptr;
            typename Test::Arguments* arguments_ptr;
            typename Test::TestResult result;

            cudaMalloc(&device_result_ptr, sizeof(typename Test::TestResult));
            cudaMalloc(&arguments_ptr, sizeof(typename Test::Arguments));
            cudaMemcpy(arguments_ptr, &arguments, sizeof(typename Test::Arguments), cudaMemcpyHostToDevice);

            interopable_test_runner_helpers::runTestViaCopy<Test><<<1,1>>>(arguments_ptr, device_result_ptr);
            cudaDeviceSynchronize();

            cudaMemcpy(&result, device_result_ptr, sizeof(typename Test::TestResult), cudaMemcpyDeviceToHost);
            cudaFree(device_result_ptr);
            cudaFree(arguments_ptr);
            
            return result;
        #else
            throw std::runtime_error("Cuda is not available to run device test");
        #endif
    }

}; /* InteropableTestRunner */


} /* namespace jcontrols_testing */

#endif /* JCONTROLS_TESTING_INTEROPABLE_TESTING_HPP_ */
