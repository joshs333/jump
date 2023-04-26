/**
 * @file threadpool.hpp
 * @author Joshua Spisak (jspisak@andrew.cmu.edu)
 * @brief Defined a threadpool that can parallelize an executor across some number of threads.
 * @version 0.1
 * @date 2023-04-26
 */
#ifndef JUMP_THREADPOOL_HPP_
#define JUMP_THREADPOOL_HPP_

// STD
#include <atomic>
#include <thread>
#include <mutex>
#include <map>
#include <vector>

namespace jump {

//! Helpers that enable threadpool_executor_interface
namespace _threadpool_executor_interface_helpers {

    //! Used to test for executor.shutdown() (positive overload)
    template <typename ExecutorT>
    constexpr auto executor_has_finish(int) -> decltype( std::declval<ExecutorT>().finish(), std::true_type{} );
    //! Used to test for executor.shutdown() (negative overload)
    template <typename>
    constexpr auto executor_has_finish(long) -> std::false_type;
    //! Used to test for executor.shutdown()
    template <typename ExecutorT>
    using executor_has_finish_test = decltype( executor_has_finish<ExecutorT>(0) );

    //! Used to test for executor.control(context) (positive overload)
    template <typename ExecutorT, typename ContextT>
    constexpr auto executor_has_control(int) -> decltype( std::declval<ExecutorT>().control(std::declval<ContextT&>()), std::true_type{} );
    //! Used to test for executor.control(context) (negative overload)
    template <typename,typename>
    constexpr auto executor_has_control(long) -> std::false_type;
    //! Used to test for executor.control(context)
    template <typename ExecutorT, typename ContextT>
    using executor_has_control_test = decltype( executor_has_control<ExecutorT, ContextT>(0) );

    //! Used to test for executor.execute(context) (positive overload)
    template <typename ExecutorT, typename ContextT>
    constexpr auto executor_has_execute(int) -> decltype( std::declval<ExecutorT>().execute(std::declval<ContextT&>()), std::true_type{} );
    //! Used to test for executor.execute(context) (negative overload)
    template <typename,typename>
    constexpr auto executor_has_execute(long) -> std::false_type;
    //! Used to test for executor.execute(context)
    template <typename ExecutorT, typename ContextT>
    using executor_has_execute_test = decltype( executor_has_execute<ExecutorT, ContextT>(0) );

} /* namespace _threadpool_executor_interface_helpers */

/**
 * @brief used to evaluate the interfacing / compatibility of a type
 *  for usage as an executor in a threadpool
 * @tparam T the type to evaluate
 */
template<typename ExecutorT, typename ContextT>
struct threadpool_executor_interface {
    /**
     * @brief determine whether the shutdown() function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool finish_defined() {
        return _threadpool_executor_interface_helpers::executor_has_finish_test<ExecutorT>::value;
    }

    /**
     * @brief determine whether the control(context) function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool control_defined() {
        return _threadpool_executor_interface_helpers::executor_has_control_test<ExecutorT, ContextT>::value;
    }

    /**
     * @brief determine whether the execute(context) function is defined
     * @return true or false if the function is defined
     */
    static constexpr bool execute_defined() {
        return _threadpool_executor_interface_helpers::executor_has_execute_test<ExecutorT, ContextT>::value;
    }

}; /* struct class_interface */

/**
 * @brief a threadpool that allows parallel execution of some executor
 */
class threadpool {
public:
    //! Context for execution that can be passed to the executor
    struct context {
        //! The signal for execution / control threads to stop
        std::atomic<bool> kill_signal_ = true;

        //! sees if the kill signal has been triggered
        bool shutdown_called() {
            return kill_signal_;
        }

        //! Triggers the kill signals
        void shutdown() {
            kill_signal_ = true;
        }
    };

    threadpool():
        executing_{false}
    {}

    /**
     * @brief Destroy the threadpool object, making
     *  sure all threads are properly stopped
     */
    ~threadpool() {
        stop();
    }

    /**
     * @brief perform parallel execution
     * @tparam executor_t the executor type to use
     * @param executor the executor to
     * @param num_threads 
     * @param block 
     */
    template<typename executor_t>
    void execute(const executor_t& executor, std::size_t num_threads, bool block = true) {
        if(executing_) {
            throw std::runtime_error("threadpool is already executing");
        }

        // ensure signals are set up correctly
        executing_ = true;
        context_.kill_signal_ = false;

        // Make sure the control thread was joined() from the previous execution (just in case it wasn't)
        if(control_thread_.joinable())
            control_thread_.join();

        control_thread_ = std::thread(&threadpool::controlThread<executor_t>, this, std::cref(executor), num_threads);
        if(block)
            control_thread_.join();
    }

    /**
     * @brief thread that actual runs control for the threadpool
     * @tparam executor_t the type of executor we are parallelizing
     * @param executor the executor to parallelize
     * @param num_threads the number of threads to parallelize over
     */
    template<typename executor_t>
    void controlThread(const executor_t& executor, std::size_t num_threads) {
        // if control(context) is defined we do flow control and dynamically create new threads
        // as the executor requests them through the control() function
        if constexpr(threadpool_executor_interface<executor_t, context>::control_defined()) {
            while(!context_.shutdown_called()) {
                // we aren't allowed to spawn any threads unless we are below the num_threads
                do {
                    {
                        std::scoped_lock sl(threads_mutex_);
                        if(dead_threads_.size() > 0) {
                            for(const auto& t_uid : dead_threads_) {
                                threads_[t_uid].join();
                                threads_.erase(t_uid);
                            }
                            dead_threads_.clear();
                        }
                    }

                    // let's avoid busy waiting
                    if(threads_.size() >= num_threads) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                } while(threads_.size() >= num_threads && !context_.shutdown_called());

                // if shutdown is called, we refuse to check spawning
                if(context_.shutdown_called())
                    break;

                // if we get here - we can spawn a thread, so we ask the executor if we should
                auto spawn = executor.control(context_);

                // if shutdown is called, we refuse to do spawn the new thread
                // (we have do not know how long executor.conrol() will take, so we check again after)
                if(context_.shutdown_called())
                    break;

                // if we spawn a thread, we register it to threads_ with a uid
                if(spawn) {
                    std::size_t new_thread_uid = thread_uid_++;
                    threads_[new_thread_uid] = std::thread(
                        &threadpool::executorThread<executor_t>, this, std::cref(executor), new_thread_uid
                    );

                // we would like to avoid the equivalent of busy waiting if the executor
                // doesn't want us to perform any work (spawn threads) over several cycles
                // we assume if there's no work to do we can take a breather
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        
        // if the control(context) function is not defined, we assume that we will just spin up
        // num_threads, and when all those threads exit the work is done :)
        } else {
            for(std::size_t new_thread_uid = 0; new_thread_uid < num_threads; ++new_thread_uid) {
                threads_[new_thread_uid] = std::thread(
                    &threadpool::executorThread<executor_t>, this, std::cref(executor), new_thread_uid
                );
            }
        }

        // Once execution completes, if the executor has a finish function, we call it to alert completion
        if constexpr(threadpool_executor_interface<executor_t, context>::finish_defined())
            executor.finish();

        // the control thread owns all the executor threads, refuse to exit unless
        // all executors have 
        for(auto& [uid, thread] : threads_) {
            thread.join();
        }
        threads_.clear();
        executing_ = false;
    }

    /**
     * @brief the function that is used for a thread to actually
     *  perform work with an executor
     * @tparam executor_t the type of executor to parallelize over
     * @param executor the executor to actually parallelize
     * @param thread_uid the uid of this thread (used to coordinate thread lifetime)
     */
    template<typename executor_t>
    void executorThread(const executor_t& executor, std::size_t thread_uid) {
        // I handle some aspects of a threads lifetime (interacting with the control thread)
        // and actually calling the kernel
        executor.execute(context_);

        std::scoped_lock sl(threads_mutex_);
        dead_threads_.push_back(thread_uid);
    }

    /**
     * @brief trigger a shutdown of the threadpool (non-blocking)
     */
    void shutdown() {
        context_.shutdown();
    }

    /**
     * @brief waits for the threadpool to stop execution (blocking)
     * @note does not trigger the threadpool to stop!
     */
    void wait() {
        if(control_thread_.joinable())
            control_thread_.join();
    }

    /**
     * @brief properly stop the threadpool (blocking until all execution is done)
     * @note calls shutdown() then wait() lol
     */
    void stop() {
        shutdown();
        wait();
    }

    /**
     * @brief allows access to the threadpool context
     * @return the context by reference
     */
    context& get_context() {
        return context_;
    }


    /**
     * @brief allows access to the threadpool context
     * @return the context by const reference
     */
    const context& get_context() const {
        return context_;
    }

private:
    //! The execution context for any execution
    context context_;

    //! Allows a unique ID to be generated for each executor thread we generate
    std::atomic<std::size_t> thread_uid_ = 0;
    //! atomic record of whether we are executing
    std::atomic<bool> executing_;
    //! Mutex for internal control of threads
    std::mutex threads_mutex_;
    //! Main thread that controls creation / joining of executor threads
    std::thread control_thread_;
    //! maps from thread_uid's to threads
    std::map<std::size_t, std::thread> threads_;
    //! mark threads as dead (can be cleaned up and removed)
    std::vector<std::size_t> dead_threads_;

}; /* class threadpool */

} /* namespace jump */

#endif /* JUMP_THREADPOOL_HPP_ */
