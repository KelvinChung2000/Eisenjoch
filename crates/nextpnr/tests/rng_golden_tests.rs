//! Golden-trace tests for the faithful `DeterministicRNG` port.
//!
//! The fixture `fixtures/nextpnr_deterministic_rng_golden.txt` was produced by
//! compiling `fixtures/nextpnr_deterministic_rng_golden.gen.cc` against the
//! real `common/kernel/deterministic_rng.h` from upstream YosysHQ nextpnr
//! `main` @ `4d235150`. Regenerate with:
//!
//! ```sh
//! g++ -O0 -std=c++17 -I<nextpnr>/common/kernel \
//!     nextpnr_deterministic_rng_golden.gen.cc -o gen && ./gen > golden.txt
//! ```
//! (`gen.cc` needs a stub `nextpnr_namespaces.h` defining the two namespace
//! macros as empty.)
//!
//! Every assertion here is exact. A near-miss stream is worse than an
//! obviously wrong one, because it still looks random.

use nextpnr::context::DeterministicRng;
use std::collections::HashMap;

/// The golden file is a flat list of values, sectioned by `# <name> [args]`
/// headers. Parse it into section-name -> lines.
fn load_golden() -> HashMap<String, Vec<String>> {
    let raw = include_str!("fixtures/nextpnr_deterministic_rng_golden.txt");
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut current: Option<String> = None;

    for line in raw.lines() {
        if let Some(header) = line.strip_prefix("# ") {
            current = Some(header.trim().to_string());
            sections.insert(header.trim().to_string(), Vec::new());
        } else if !line.trim().is_empty() {
            let key = current
                .as_ref()
                .expect("golden fixture has values before its first header");
            sections
                .get_mut(key)
                .expect("section registered on header")
                .push(line.trim().to_string());
        }
    }

    assert!(!sections.is_empty(), "golden fixture parsed as empty");
    sections
}

fn section<'a>(g: &'a HashMap<String, Vec<String>>, name: &str) -> &'a [String] {
    g.get(name)
        .unwrap_or_else(|| panic!("golden fixture is missing section `{name}`"))
}

#[test]
fn default_state_matches_nextpnr() {
    let golden = load_golden();
    let expected = section(&golden, "default_rng64");

    // Default construction, no seeding: nextpnr starts at 0x3141592653589793.
    let mut rng = DeterministicRng::default();
    for (i, want) in expected.iter().enumerate() {
        let want: u64 = want.parse().expect("golden value is a u64");
        assert_eq!(rng.rng64(), want, "rng64 diverged at draw {i}");
    }
    assert_eq!(expected.len(), 1000, "expected 1000 default draws");
}

#[test]
fn seeded_streams_match_nextpnr() {
    let golden = load_golden();

    // Seed 0 is the interesting one: rngseed falls back to the default state
    // rather than seeding with zero.
    for seed in [0u64, 1, 42, 12345, 0xDEADBEEF, 0x3141592653589793] {
        let expected = section(&golden, &format!("seeded_rng64 {seed}"));
        let mut rng = DeterministicRng::new(seed);
        for (i, want) in expected.iter().enumerate() {
            let want: u64 = want.parse().expect("golden value is a u64");
            assert_eq!(rng.rng64(), want, "seed {seed} diverged at draw {i}");
        }
    }
}

#[test]
fn seed_zero_is_not_seed_one() {
    // Guards the fallback: a zero seed must land on the default state, not on
    // some clamped-to-1 state.
    let mut zero = DeterministicRng::new(0);
    let mut one = DeterministicRng::new(1);
    assert_ne!(zero.rng64(), one.rng64());

    let mut zero = DeterministicRng::new(0);
    let mut explicit = DeterministicRng::new(0x3141592653589793);
    assert_eq!(
        zero.rng64(),
        explicit.rng64(),
        "a zero seed must be equivalent to seeding with the default state"
    );
}

#[test]
fn rng_30bit_matches_nextpnr() {
    let golden = load_golden();
    let expected = section(&golden, "rng_30bit");

    let mut rng = DeterministicRng::new(42);
    for (i, want) in expected.iter().enumerate() {
        let want: i32 = want.parse().expect("golden value is an i32");
        let got = rng.rng();
        assert_eq!(got, want, "rng() diverged at draw {i}");
        assert!((0..=0x3fff_ffff).contains(&got), "rng() left its 30-bit range");
    }
}

#[test]
fn rng_n_matches_nextpnr() {
    let golden = load_golden();

    // These n straddle power-of-two boundaries, where the rejection loop
    // actually fires and a modulo implementation would visibly diverge.
    for n in [1i32, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 65537] {
        let expected = section(&golden, &format!("rng_n {n}"));
        let mut rng = DeterministicRng::new(42);
        for (i, want) in expected.iter().enumerate() {
            let want: i32 = want.parse().expect("golden value is an i32");
            let got = rng.rng_n(n);
            assert_eq!(got, want, "rng_n({n}) diverged at draw {i}");
            assert!((0..n).contains(&got), "rng_n({n}) returned out-of-range {got}");
        }
    }
}

#[test]
fn rngf_matches_nextpnr_bit_for_bit() {
    let golden = load_golden();

    // Hex floats, compared as exact bits: rngf is computed entirely in f32,
    // and doing any of it in f64 would drift here.
    for n in [1.0f32, 2.5, 100.0] {
        let key = format!("rngf {}", format_hex_f64(n as f64));
        let expected = section(&golden, &key);
        let mut rng = DeterministicRng::new(42);
        for (i, want) in expected.iter().enumerate() {
            let want = parse_hex_f64(want) as f32;
            let got = rng.rngf(n);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "rngf({n}) diverged at draw {i}: got {got:?}, want {want:?}"
            );
        }
    }
}

#[test]
fn shuffle_matches_nextpnr() {
    let golden = load_golden();

    for seed in [1u64, 42, 12345] {
        for len in [2usize, 5, 16, 32, 33] {
            let expected = section(&golden, &format!("shuffle {seed} {len}"));
            let mut rng = DeterministicRng::new(seed);
            let mut v: Vec<i32> = (0..len as i32).collect();
            rng.shuffle(&mut v);

            let want: Vec<i32> = expected
                .iter()
                .map(|s| s.parse().expect("golden value is an i32"))
                .collect();
            assert_eq!(v, want, "shuffle(seed={seed}, len={len}) diverged");
        }
    }
}

#[test]
fn sorted_shuffle_matches_nextpnr() {
    let golden = load_golden();

    for seed in [1u64, 42] {
        let expected = section(&golden, &format!("sorted_shuffle {seed}"));
        let mut rng = DeterministicRng::new(seed);
        let mut v = vec![9, 3, 7, 1, 8, 2, 6, 0, 5, 4];
        rng.sorted_shuffle(&mut v);

        let want: Vec<i32> = expected
            .iter()
            .map(|s| s.parse().expect("golden value is an i32"))
            .collect();
        assert_eq!(v, want, "sorted_shuffle(seed={seed}) diverged");
    }
}

#[test]
fn sorted_shuffle_ignores_input_order() {
    // It sorts first, so two permutations of the same multiset must agree.
    let mut a = DeterministicRng::new(7);
    let mut v_a = vec![5, 1, 4, 2, 3];
    a.sorted_shuffle(&mut v_a);

    let mut b = DeterministicRng::new(7);
    let mut v_b = vec![3, 2, 4, 1, 5];
    b.sorted_shuffle(&mut v_b);

    assert_eq!(v_a, v_b);
}

#[test]
#[should_panic(expected = "rng_n requires n > 0")]
fn rng_n_rejects_zero() {
    DeterministicRng::new(1).rng_n(0);
}

#[test]
#[should_panic(expected = "rng_n requires n > 0")]
fn rng_n_rejects_negative() {
    DeterministicRng::new(1).rng_n(-5);
}

#[test]
fn aliases_delegate_to_the_faithful_stream() {
    // The compatibility aliases must not introduce a second stream.
    let mut a = DeterministicRng::new(42);
    let mut b = DeterministicRng::new(42);
    assert_eq!(a.next_u64(), b.rng64());

    let mut a = DeterministicRng::new(42);
    let mut b = DeterministicRng::new(42);
    assert_eq!(a.next_u32(), b.rng64() as u32);

    // next_range must use the unbiased sampler, not a modulo.
    let mut a = DeterministicRng::new(42);
    let mut b = DeterministicRng::new(42);
    assert_eq!(a.next_range(100), b.rng_n(100) as u32);
}

// ---------------------------------------------------------------------------
// Hex-float helpers. Rust has no `%a` parser in std, and the values must round
// trip exactly, so parse the C99 form by hand.
// ---------------------------------------------------------------------------

fn format_hex_f64(v: f64) -> String {
    // Only used to rebuild the section keys the C++ `%a` emitted, for the
    // handful of constants in the fixture.
    match v {
        x if x == 1.0 => "0x1p+0".to_string(),
        x if x == 2.5 => "0x1.4p+1".to_string(),
        x if x == 100.0 => "0x1.9p+6".to_string(),
        other => panic!("no known %a spelling for {other}"),
    }
}

fn parse_hex_f64(s: &str) -> f64 {
    let s = s.trim();
    if s == "0x0p+0" || s == "-0x0p+0" {
        return 0.0;
    }
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let s = s
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("not a C99 hex float: {s}"));
    let (mantissa, exp) = s
        .split_once('p')
        .unwrap_or_else(|| panic!("hex float has no exponent: {s}"));
    let exp: i32 = exp.parse().expect("hex float exponent is an integer");

    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };

    let mut value = u64::from_str_radix(int_part, 16).expect("hex float integer part") as f64;
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        let digit = c.to_digit(16).expect("hex float fraction digit") as f64;
        value += digit * scale;
        scale /= 16.0;
    }

    let out = value * 2f64.powi(exp);
    if neg { -out } else { out }
}
