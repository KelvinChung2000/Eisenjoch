// Golden-trace generator for nextpnr HeAP's EquationSystem<double>.
// The struct body is copied verbatim from upstream YosysHQ nextpnr main @ 4d235150,
// common/place/placer_heap.cc, so this exercises the real Eigen solve path
// (ConjugateGradient<SparseMatrix<double>, Lower|Upper>, i.e. Jacobi-preconditioned CG,
// via solveWithGuess).
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <cassert>
#include <cstdio>
#include <vector>

template <typename T> struct EquationSystem
{
    EquationSystem(size_t rows, size_t cols)
    {
        A.resize(cols);
        rhs.resize(rows);
    }

    std::vector<std::vector<std::pair<int, T>>> A;
    std::vector<T> rhs;
    void reset()
    {
        for (auto &col : A)
            col.clear();
        std::fill(rhs.begin(), rhs.end(), T());
    }

    void add_coeff(int row, int col, T val)
    {
        auto &Ac = A.at(col);
        int b = 0, e = int(Ac.size()) - 1;
        while (b <= e) {
            int i = (b + e) / 2;
            if (Ac.at(i).first == row) {
                Ac.at(i).second += val;
                return;
            }
            if (Ac.at(i).first > row)
                e = i - 1;
            else
                b = i + 1;
        }
        Ac.insert(Ac.begin() + b, std::make_pair(row, val));
    }

    void add_rhs(int row, T val) { rhs[row] += val; }

    void solve(std::vector<T> &x, float tolerance)
    {
        using namespace Eigen;
        if (x.empty())
            return;
        assert(x.size() == A.size());

        VectorXd vx(x.size()), vb(rhs.size());
        SparseMatrix<T> mat(A.size(), A.size());

        std::vector<int> colnnz;
        for (auto &Ac : A)
            colnnz.push_back(int(Ac.size()));
        mat.reserve(colnnz);
        for (int col = 0; col < int(A.size()); col++) {
            auto &Ac = A.at(col);
            for (auto &el : Ac)
                mat.insert(el.first, col) = el.second;
        }

        for (int i = 0; i < int(x.size()); i++)
            vx[i] = x.at(i);
        for (int i = 0; i < int(rhs.size()); i++)
            vb[i] = rhs.at(i);

        ConjugateGradient<SparseMatrix<T>, Lower | Upper> solver;
        solver.setTolerance(tolerance);
        VectorXd xr = solver.compute(mat).solveWithGuess(vb, vx);
        for (int i = 0; i < int(x.size()); i++)
            x.at(i) = xr[i];
    }
};

// Deterministic pseudo-random source so the Rust side can rebuild the exact
// same systems without shipping the matrices themselves.
static unsigned long long st = 0x12345678ABCDEF01ull;
static double frand()
{
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return double(st % 1000) / 100.0;
}

// Build an SPD system the way HeAP does: symmetric off-diagonal pairs plus a
// dominant diagonal, which is what the bound2bound net model produces.
static void build(EquationSystem<double> &es, int n, int band)
{
    for (int i = 0; i < n; i++) {
        double diag = 0;
        for (int d = 1; d <= band; d++) {
            int j = i + d;
            if (j >= n)
                continue;
            double w = frand() + 0.5;
            es.add_coeff(i, j, -w);
            es.add_coeff(j, i, -w);
            diag += w;
        }
        es.add_coeff(i, i, diag + 1.0);
        es.add_rhs(i, frand());
    }
}

int main()
{
    struct { int n, band; float tol; } cases[] = {
        {1, 0, 1e-5f},
        {2, 1, 1e-5f},
        {5, 1, 1e-5f},
        {10, 2, 1e-5f},
        {32, 3, 1e-5f},
        {64, 4, 1e-6f},
        {100, 5, 1e-7f},
        {32, 3, 1e-13f},
        {64, 4, 1e-13f},
    };

    for (auto &c : cases) {
        st = 0x12345678ABCDEF01ull; // reset the stream per case
        EquationSystem<double> es(c.n, c.n);
        build(es, c.n, c.band);

        std::vector<double> x(c.n, 0.0);
        es.solve(x, c.tol);

        printf("# solve %d %d %g\n", c.n, c.band, (double)c.tol);
        for (int i = 0; i < c.n; i++)
            printf("%a\n", x[i]);
    }

    // A non-zero initial guess exercises solveWithGuess properly: CG starting
    // from a warm x must land on the same solution.
    {
        st = 0x12345678ABCDEF01ull;
        EquationSystem<double> es(20, 20);
        build(es, 20, 3);
        std::vector<double> x(20);
        for (int i = 0; i < 20; i++)
            x[i] = 0.25 * i;
        es.solve(x, 1e-6f);
        printf("# solve_guess 20 3 1e-06\n");
        for (int i = 0; i < 20; i++)
            printf("%a\n", x[i]);
    }

    // add_coeff must accumulate on repeat, and reset() must clear both A and rhs.
    {
        EquationSystem<double> es(3, 3);
        es.add_coeff(0, 0, 2.0);
        es.add_coeff(0, 0, 3.0); // accumulates to 5
        es.add_coeff(2, 0, 1.0);
        es.add_coeff(1, 0, 4.0); // inserted between, keeping rows sorted
        printf("# accumulate\n");
        for (auto &el : es.A[0])
            printf("%d %a\n", el.first, el.second);
    }

    return 0;
}
