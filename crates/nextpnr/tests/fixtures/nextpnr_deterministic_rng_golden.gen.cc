// Golden-trace generator for nextpnr's DeterministicRNG.
// Compiled against upstream nextpnr main @ 4d235150 common/kernel/deterministic_rng.h.
// Output is consumed as a test fixture by eisenjoch's faithful RNG port.
#include "deterministic_rng.h"
#include <cstdio>
#include <numeric>

int main()
{
    // 1. Default-constructed state, first 1000 rng64() draws.
    {
        DeterministicRNG r;
        printf("# default_rng64\n");
        for (int i = 0; i < 1000; i++)
            printf("%llu\n", (unsigned long long)r.rng64());
    }

    // 2. Seeded streams: rng64() after rngseed().
    for (uint64_t seed : {0ull, 1ull, 42ull, 12345ull, 0xDEADBEEFull, 0x3141592653589793ull}) {
        DeterministicRNG r;
        r.rngseed(seed);
        printf("# seeded_rng64 %llu\n", (unsigned long long)seed);
        for (int i = 0; i < 100; i++)
            printf("%llu\n", (unsigned long long)r.rng64());
    }

    // 3. rng() -- the 30-bit variant.
    {
        DeterministicRNG r;
        r.rngseed(42);
        printf("# rng_30bit\n");
        for (int i = 0; i < 100; i++)
            printf("%d\n", r.rng());
    }

    // 4. rng(n) -- rejection sampling against the next power of two.
    //    n values chosen to straddle power-of-two boundaries where the
    //    rejection loop actually bites.
    for (int n : {1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 65537}) {
        DeterministicRNG r;
        r.rngseed(42);
        printf("# rng_n %d\n", n);
        for (int i = 0; i < 100; i++)
            printf("%d\n", r.rng(n));
    }

    // 5. rngf(n) -- float variant. Hex float so the comparison is exact.
    for (float n : {1.0f, 2.5f, 100.0f}) {
        DeterministicRNG r;
        r.rngseed(42);
        printf("# rngf %a\n", (double)n);
        for (int i = 0; i < 50; i++)
            printf("%a\n", (double)r.rngf(n));
    }

    // 6. shuffle() -- the exact permutation matters, and it consumes the
    //    stream in a specific order.
    for (uint64_t seed : {1ull, 42ull, 12345ull}) {
        for (int len : {2, 5, 16, 32, 33}) {
            DeterministicRNG r;
            r.rngseed(seed);
            std::vector<int> v(len);
            std::iota(v.begin(), v.end(), 0);
            r.shuffle(v);
            printf("# shuffle %llu %d\n", (unsigned long long)seed, len);
            for (int i = 0; i < len; i++)
                printf("%d\n", v[i]);
        }
    }

    // 7. sorted_shuffle() -- sorts first, so it is order-independent on input.
    for (uint64_t seed : {1ull, 42ull}) {
        DeterministicRNG r;
        r.rngseed(seed);
        std::vector<int> v = {9, 3, 7, 1, 8, 2, 6, 0, 5, 4};
        r.sorted_shuffle(v);
        printf("# sorted_shuffle %llu\n", (unsigned long long)seed);
        for (size_t i = 0; i < v.size(); i++)
            printf("%d\n", v[i]);
    }

    return 0;
}
