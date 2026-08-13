// Golden-trace generator for nextpnr's ConstraintLegaliseWorker::IncreasingDiameterSearch.
// The class body below is copied verbatim from upstream YosysHQ nextpnr main @ 4d235150,
// common/place/place_common.cc, so the trace is the real traversal order.
#include <algorithm>
#include <cstdio>

class IncreasingDiameterSearch
{
  public:
    IncreasingDiameterSearch() : start(0), min(0), max(-1) {};
    IncreasingDiameterSearch(int x) : start(x), min(x), max(x) {};
    IncreasingDiameterSearch(int start, int min, int max) : start(start), min(min), max(max) {};
    bool done() const { return (diameter > (max - min)); };
    int get() const
    {
        int val = start + sign * diameter;
        val = std::max(val, min);
        val = std::min(val, max);
        return val;
    }

    void next()
    {
        if (sign == 0) {
            sign = 1;
            diameter = 1;
        } else if (sign == -1) {
            sign = 1;
            if ((start + sign * diameter) > max)
                sign = -1;
            ++diameter;
        } else {
            sign = -1;
            if ((start + sign * diameter) < min) {
                sign = 1;
                ++diameter;
            }
        }
    }

    void reset()
    {
        sign = 0;
        diameter = 0;
    }

  private:
    int start, min, max;
    int diameter = 0;
    int sign = 0;
};

int main()
{
    // Default construction is already exhausted (max < min).
    {
        IncreasingDiameterSearch s;
        printf("# default_done %d\n", (int)s.done());
    }

    // Single-value construction.
    {
        IncreasingDiameterSearch s(7);
        printf("# single 7\n");
        int guard = 0;
        while (!s.done() && guard++ < 50)
            { printf("%d\n", s.get()); s.next(); }
    }

    // Ranged construction, including starts at and past the edges, which is
    // where the sign flipping actually matters.
    struct { int start, min, max; } cases[] = {
        {5, 0, 10},   // centred
        {0, 0, 10},   // at the low edge
        {10, 0, 10},  // at the high edge
        {2, 0, 10},   // off-centre low
        {8, 0, 10},   // off-centre high
        {3, 3, 3},    // degenerate
        {0, 0, 1},    // two values
        {4, 2, 9},    // non-zero min
    };

    for (auto &c : cases) {
        IncreasingDiameterSearch s(c.start, c.min, c.max);
        printf("# range %d %d %d\n", c.start, c.min, c.max);
        int guard = 0;
        while (!s.done() && guard++ < 200)
            { printf("%d\n", s.get()); s.next(); }
    }

    // reset() mid-traversal must restart from the beginning.
    {
        IncreasingDiameterSearch s(5, 0, 10);
        printf("# reset 5 0 10\n");
        for (int i = 0; i < 4; i++) { printf("%d\n", s.get()); s.next(); }
        s.reset();
        int guard = 0;
        while (!s.done() && guard++ < 200)
            { printf("%d\n", s.get()); s.next(); }
    }

    return 0;
}
