/**
 * @brief I had some weird issues with the indices and the modulo operator
 *  running on the GPU... at a certain point I should try to run that down.
 *  It might be a compiler issue, supposed I'm not totally insane / stupid,
 *  which I never count outside the realm of possibility.
 */

// struct indices {

//     std::size_t values_[4];
// }

#if JUMP_ENABLE_CUDA
__global__
void testThing(jump::indices range) {
    auto z = jump::indices::zero(range.dims());
    z[0] = 20;
    z %= range;
    std::printf("%lu\n", z[0]);
}

TEST(TEST_SUITE_NAME, multiIndices1DModulo) {
    auto range = jump::indices::zero(1);
    range[0] = 50;
    testThing<<<1,1>>>(range);
    cudaDeviceSynchronize();
}
#endif

int main(int argc, char** argv) {

    return 0;
}