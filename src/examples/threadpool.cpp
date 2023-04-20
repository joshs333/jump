#include <jump/threadpool.hpp>
#include <cstdio>
#include <atomic>
#include <thread>
#include <mutex>
#include <map>
#include <vector>
#include <chrono>
#include <queue>
#include <memory>
#include <functional>


// The form of an executor
template<typename T>
class queue_executor {
public:
    queue_executor(
        std::queue<T>& queue,
        std::function<void(T)> function
    ):
        queue_(&queue),
        function_(function)
    {}

    bool control(jump::threadpool::context& context) const {
        std::scoped_lock l(queue_mutex_);
        if(queue_->size() > 0)
            return true;
        context.shutdown();
        return false;
    }

    /// Shutdown can be called at any point during this execution
    void execute(jump::threadpool::context& context) const {
        if(context.shutdown_called()) return;
        queue_mutex_.lock();
        if(queue_->size() <= 0) { queue_mutex_.unlock(); return; }
        auto v = queue_->front();
        queue_->pop();
        queue_mutex_.unlock();

        function_(v);
    }

private:
    std::queue<int>* queue_;
    std::function<void(T)> function_;
    mutable std::mutex queue_mutex_;

};


int main(int argc, char** argv) {
    std::printf("Threadpool example!\n");


    auto queue = std::make_shared<std::queue<int>>();

    for(int i = 0; i < 100; ++i) {
        queue->push(i);
    }

    jump::threadpool pool;
    pool.execute(queue_executor<int>(*queue, [&](int v){ 
        std::printf("%d\n", v);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }), 100);

    return 0;
}
