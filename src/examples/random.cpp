#include <jump/random.hpp>
#include <jump/parallel.hpp>

#include <cstdint>
#include <cstdio>
#include <random>
#include <chrono>

#include <curand.h>
#include <curand_kernel.h>

struct normal_kernel {
    static const bool device_compatible = true;
    static const bool host_compatible = false;

    JUMP_INTEROPABLE
    void kernel(const std::size_t& idx) const {
        #if JUMP_ON_DEVICE
            curandState s;
            curand_init(0, 0, idx, &s);
            for(int i = 0; i < vals.size(); ++i) {
                vals[i] = curand_normal(&s);
            }
        #endif
    }

    void to_device() {
        vals.to_device();
    }

    void from_device() {
        vals.from_device();
    }

    jump::array<double> vals;
}; /* struct multiplier_kernel */

int main(int argc, char** argv) {
    std::printf("Hello you\n");

    std::size_t nvals = 1000;

    {
        jump::prng rng;
        rng.seed(jump::random_seed());
        jump::normal_distribution nd(1.0, 2.0);
        std::normal_distribution snd(1.0, 2.0);

        std::vector<double> values;
        double avg = 0.0;
        for(int i = 0; i < nvals; ++i) {
            // auto r = nd(rng);
            auto r = snd(rng);
            avg += r;
            values.push_back(r);
        }
        std::vector<double> stddevs;
        avg = avg / values.size();
        double stddev = 0;
        for(std::size_t i = 0; i < values.size(); ++i) {
            auto v = std::pow(values[i] - avg, 2.0);
            stddev += v;
            stddevs.push_back(v);
        }
        stddev = std::sqrt(stddev / (values.size() - 1));
        std::printf("%3.2f\n", avg);
        std::printf("%3.2f\n", stddev);
    }

    {
        jump::array<double> values(nvals, 0);

        jump::iterate(1, normal_kernel{values}, {.target = jump::par::cuda});

        double avg = 0;
        for(std::size_t i = 0; i < values.size(); ++i) {
            avg += values[i];
        }
        std::vector<double> stddevs;
        avg = avg / values.size();
        double stddev = 0;
        for(std::size_t i = 0; i < values.size(); ++i) {
            auto v = std::pow(values[i] - avg, 2.0);
            stddev += v;
            stddevs.push_back(v);
        }
        stddev = std::sqrt(stddev / (values.size() - 1));
        std::printf("%3.2f\n", avg);
        std::printf("%3.2f\n", stddev);

    }
    return 0;
}

