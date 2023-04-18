#include <cstdio>

// https://eli.thegreenplace.net/2016/c11-threads-affinity-and-hyperthreading/#footnote-reference-1

namespace jump {

class threadpool {
    struct context {
        std::atomic<std::size_t> active_threads_;

        std::atomic<std::size_t> thread_id_;

        bool should_stop() {
        }

        void shutdown() {
        }

        void set_affinity(std::size_t core_number) {
        }
    }

    template<typename kernel_t>
    void execute(const kernel_t& kernel, ) {
        // I spawn the control thread and 
    }

    void controlThread() {
        // I handle thread lifetime
    }

    void executorThread() {
        // I handle some aspects of a threads lifetime (interacting with the control thread)
        // and actually calling the kernel
    }
};

}

struct queue_executor {

};

struct executor {

    bool spawn(jump::threadpool::context& context) const {

    }

    void execute(jump::threadpool::context& context) const {

    }

};


int main(int argc, char** argv) {
    std::printf("Threadpool example!\n");

    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::threadpool, .cores = {1, 2, 3}});
    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::cuda, .cores = {1, 2, 3}});
    // jump::iterate(arr.shape(), {0, 1}, kernel{arr}, {.target = jump::par::seq, .cores = {1, 2, 3}});
    return 0;
}
