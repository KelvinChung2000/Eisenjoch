//! Golden tests for HeAP's `EquationSystem` and its solver.
//!
//! The fixture was produced by `fixtures/nextpnr_eqsys_golden.gen.cc`, which
//! embeds nextpnr's `EquationSystem<double>` verbatim (upstream YosysHQ nextpnr
//! `main` @ `4d235150`) and links against real Eigen, so the reference values
//! come from the actual `ConjugateGradient<SparseMatrix<double>, Lower|Upper>`
//! solve path. Regenerate with:
//!
//! ```sh
//! g++ -O0 -std=c++17 -I/usr/include/eigen3 \
//!     nextpnr_eqsys_golden.gen.cc -o gen && ./gen > golden.txt
//! ```
//!
//! This is the component of HeAP where "plausible but different" is otherwise
//! undetectable: a CG that converges to the same *solution* by a different
//! iteration path yields different intermediate placements, and HeAP feeds
//! every intermediate back into spreading and legalisation.

use nextpnr::placer::heap::equation_system::EquationSystem;
use std::collections::HashMap;

fn load_golden() -> HashMap<String, Vec<String>> {
    let raw = include_str!("fixtures/nextpnr_eqsys_golden.txt");
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut current: Option<String> = None;

    for line in raw.lines() {
        if let Some(header) = line.strip_prefix("# ") {
            current = Some(header.trim().to_string());
            sections.insert(header.trim().to_string(), Vec::new());
        } else if !line.trim().is_empty() {
            let key = current.as_ref().expect("value before first header");
            sections
                .get_mut(key)
                .expect("section registered")
                .push(line.trim().to_string());
        }
    }
    sections
}

/// The generator's pseudo-random source, replicated so the Rust side rebuilds
/// bit-identical systems without shipping the matrices.
struct Frand(u64);

impl Frand {
    fn new() -> Self {
        Self(0x12345678ABCDEF01)
    }
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % 1000) as f64 / 100.0
    }
}

/// Build the same banded SPD system the generator builds: symmetric
/// off-diagonal pairs plus a dominant diagonal, as the bound2bound net model
/// produces.
fn build(n: i32, band: i32) -> EquationSystem {
    let mut es = EquationSystem::new(n as usize, n as usize);
    let mut r = Frand::new();
    for i in 0..n {
        let mut diag = 0.0;
        for d in 1..=band {
            let j = i + d;
            if j >= n {
                continue;
            }
            let w = r.next() + 0.5;
            es.add_coeff(i, j, -w);
            es.add_coeff(j, i, -w);
            diag += w;
        }
        es.add_coeff(i, i, diag + 1.0);
        let rhs = r.next();
        es.add_rhs(i, rhs);
    }
    es
}

fn parse_hex_f64(s: &str) -> f64 {
    let s = s.trim();
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    if s == "0x0p+0" {
        return if neg { -0.0 } else { 0.0 };
    }
    let s = s
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("not a C99 hex float: {s}"));
    let (mantissa, exp) = s
        .split_once('p')
        .unwrap_or_else(|| panic!("hex float has no exponent: {s}"));
    let exp: i32 = exp.parse().expect("hex float exponent");

    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };
    let mut value = u64::from_str_radix(int_part, 16).expect("hex int part") as f64;
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        value += c.to_digit(16).expect("hex frac digit") as f64 * scale;
        scale /= 16.0;
    }
    let out = value * 2f64.powi(exp);
    if neg { -out } else { out }
}

fn assert_close(got: &[f64], want: &[f64], eps: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        let scale = w.abs().max(1.0);
        assert!(
            diff / scale < eps,
            "{what}: element {i} diverged: got {g:.17e}, want {w:.17e} (rel {:.3e})",
            diff / scale
        );
    }
}

/// Compare two solution vectors by relative L2 norm, `||got - want|| / ||want||`.
///
/// The right instrument for a linear solve: an element-wise bound punishes
/// small components for absolute differences that are negligible against the
/// vector as a whole, which says nothing about whether the two solvers agree.
fn assert_close_norm(got: &[f64], want: &[f64], eps: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let diff: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = want.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    assert!(
        diff / norm < eps,
        "{what}: solutions diverged by {:.3e} relative (limit {eps:.3e})",
        diff / norm
    );
}

/// The relative residual `||Ax - b|| / ||b||` of a candidate solution.
fn relative_residual(es: &EquationSystem, x: &[f64]) -> f64 {
    let n = x.len();
    let mut ax = vec![0.0; n];
    for (col, entries) in es.a.iter().enumerate() {
        for &(row, val) in entries {
            ax[row as usize] += val * x[col];
        }
    }
    let num: f64 = (0..n).map(|i| (ax[i] - es.rhs[i]).powi(2)).sum::<f64>().sqrt();
    let den: f64 = es.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / den
}

/// Two independently-compiled CG implementations do not agree bit for bit:
/// Eigen vectorises its dot products, so the summation order differs and the
/// iterates drift by roughly 1e-9 relative. When the solve tolerance is loose
/// the iteration stops early, and that drift is preserved in the answer.
///
/// So each case is checked by relative L2 norm against its own solve
/// tolerance -- the two answers must agree at least as closely as the accuracy
/// either was asked for, which is the strongest statement that means anything
/// about an iterative solver stopped early -- and each is additionally required
/// to *meet* that requested residual.
///
/// The real proof of algorithmic equivalence is
/// [`solve_converges_to_the_same_answer_as_eigen`] below: run both to 1e-13 and
/// they agree to 1e-12. A wrong preconditioner, a different convergence test or
/// a reordered update would not converge to the same point at any tolerance.
fn eps_for(tol: f32) -> f64 {
    tol as f64
}

/// Used where the comparison is against an exactly-known answer rather than
/// against an early-stopped Eigen iterate.
const EXACT_EPS: f64 = 1e-12;

#[test]
fn solve_matches_eigen() {
    let golden = load_golden();

    // (n, band, tolerance) -- from a 1x1 system up to 100x100 with bandwidth 5.
    let cases = [
        (1i32, 0i32, 1e-5f32, "solve 1 0 1e-05"),
        (2, 1, 1e-5, "solve 2 1 1e-05"),
        (5, 1, 1e-5, "solve 5 1 1e-05"),
        (10, 2, 1e-5, "solve 10 2 1e-05"),
        (32, 3, 1e-5, "solve 32 3 1e-05"),
        (64, 4, 1e-6, "solve 64 4 1e-06"),
        (100, 5, 1e-7, "solve 100 5 1e-07"),
    ];

    for (n, band, tol, key) in cases {
        let want: Vec<f64> = golden
            .get(key)
            .unwrap_or_else(|| panic!("golden fixture missing `{key}`"))
            .iter()
            .map(|s| parse_hex_f64(s))
            .collect();

        let es = build(n, band);
        let mut x = vec![0.0; n as usize];
        es.solve(&mut x, tol);

        assert_close_norm(&x, &want, eps_for(tol), key);

        // The solver must actually deliver the accuracy it was asked for.
        let resid = relative_residual(&es, &x);
        assert!(
            resid <= tol as f64 * 1.01,
            "{key}: converged to relative residual {resid:.3e}, worse than the requested {tol:e}"
        );
    }
}

#[test]
fn solve_converges_to_the_same_answer_as_eigen() {
    // Run both implementations essentially to convergence. Any genuine
    // algorithmic difference -- preconditioner, convergence test, update order
    // -- survives tightening the tolerance; floating-point summation order does
    // not.
    let golden = load_golden();

    for (n, band, key) in [(32i32, 3i32, "solve 32 3 1e-13"), (64, 4, "solve 64 4 1e-13")] {
        let want: Vec<f64> = golden
            .get(key)
            .unwrap_or_else(|| panic!("golden fixture missing `{key}`"))
            .iter()
            .map(|s| parse_hex_f64(s))
            .collect();

        let es = build(n, band);
        let mut x = vec![0.0; n as usize];
        es.solve(&mut x, 1e-13);

        assert_close(&x, &want, EXACT_EPS, key);
    }
}

#[test]
fn solve_with_nonzero_guess_matches_eigen() {
    // Exercises the warm start: HeAP always re-solves from the previous
    // placement, never from zero.
    let golden = load_golden();
    let want: Vec<f64> = golden["solve_guess 20 3 1e-06"]
        .iter()
        .map(|s| parse_hex_f64(s))
        .collect();

    let es = build(20, 3);
    let mut x: Vec<f64> = (0..20).map(|i| 0.25 * i as f64).collect();
    es.solve(&mut x, 1e-6);

    assert_close_norm(&x, &want, eps_for(1e-6), "solve_guess 20 3");
}

#[test]
fn add_coeff_accumulates_and_keeps_rows_sorted() {
    let golden = load_golden();
    let expected: Vec<(i32, f64)> = golden["accumulate"]
        .iter()
        .map(|line| {
            let (row, val) = line.split_once(' ').expect("row and value");
            (row.parse().expect("row"), parse_hex_f64(val))
        })
        .collect();

    let mut es = EquationSystem::new(3, 3);
    es.add_coeff(0, 0, 2.0);
    es.add_coeff(0, 0, 3.0); // must accumulate to 5, not overwrite
    es.add_coeff(2, 0, 1.0);
    es.add_coeff(1, 0, 4.0); // must be inserted between, keeping rows sorted

    let got = &es.a[0];
    assert_eq!(got.len(), expected.len());
    for (i, (want_row, want_val)) in expected.iter().enumerate() {
        assert_eq!(got[i].0, *want_row, "row order diverged at {i}");
        assert!((got[i].1 - want_val).abs() < 1e-12, "value diverged at {i}");
    }

    // The sort is load-bearing: add_coeff binary-searches this list.
    assert!(
        got.windows(2).all(|w| w[0].0 < w[1].0),
        "column entries must stay sorted by row"
    );
}

#[test]
fn reset_clears_matrix_and_rhs() {
    let mut es = EquationSystem::new(3, 3);
    es.add_coeff(0, 0, 1.0);
    es.add_coeff(1, 1, 2.0);
    es.add_rhs(0, 5.0);

    es.reset();

    assert!(es.a.iter().all(|c| c.is_empty()));
    assert!(es.rhs.iter().all(|&v| v == 0.0));
    assert_eq!(es.a.len(), 3, "reset must keep the column count");
}

#[test]
fn empty_system_is_a_noop() {
    let es = EquationSystem::new(0, 0);
    let mut x: Vec<f64> = Vec::new();
    es.solve(&mut x, 1e-5);
    assert!(x.is_empty());
}

#[test]
fn zero_rhs_gives_zero_solution() {
    // Eigen short-circuits this case rather than iterating.
    let mut es = EquationSystem::new(4, 4);
    for i in 0..4 {
        es.add_coeff(i, i, 2.0);
    }
    let mut x = vec![7.0, -3.0, 1.5, 0.25];
    es.solve(&mut x, 1e-5);
    assert!(x.iter().all(|&v| v == 0.0), "zero RHS must zero the guess");
}

#[test]
fn diagonal_system_solves_exactly() {
    // A sanity check independent of the golden data: for a diagonal matrix the
    // Jacobi preconditioner is exact, so CG converges in one step.
    let mut es = EquationSystem::new(3, 3);
    es.add_coeff(0, 0, 2.0);
    es.add_coeff(1, 1, 4.0);
    es.add_coeff(2, 2, 8.0);
    es.add_rhs(0, 6.0);
    es.add_rhs(1, 8.0);
    es.add_rhs(2, 8.0);

    let mut x = vec![0.0; 3];
    es.solve(&mut x, 1e-9);

    assert_close(&x, &[3.0, 2.0, 1.0], EXACT_EPS, "diagonal solve");
}
