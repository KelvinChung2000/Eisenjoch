//! Tests for the faithful `place_common` port.
//!
//! `IncreasingDiameterSearch` is checked against a golden trace produced by the
//! real C++ class, copied verbatim into
//! `fixtures/nextpnr_ids_golden.gen.cc` from upstream YosysHQ nextpnr `main`
//! @ `4d235150`.
//!
//! This one is worth pinning: the traversal emits *clamped duplicates* once one
//! side of the range is exhausted (`5 6 4 7 3 8 2 9 1 10 0 10 10 10 10 10` for
//! `start=5, min=0, max=10`). A tidier implementation that skipped the repeats
//! would silently change how many candidate locations the constraint legaliser
//! evaluates before giving up.

use nextpnr::placer::place_common::IncreasingDiameterSearch;
use std::collections::HashMap;

fn load_golden() -> HashMap<String, Vec<i32>> {
    let raw = include_str!("fixtures/nextpnr_ids_golden.txt");
    let mut sections: HashMap<String, Vec<i32>> = HashMap::new();
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
                .push(line.trim().parse().expect("golden value is an i32"));
        }
    }
    sections
}

/// Walk a search to exhaustion, with the same guard the generator used so a
/// non-terminating port fails loudly instead of hanging.
fn walk(mut s: IncreasingDiameterSearch, guard_limit: usize) -> Vec<i32> {
    let mut out = Vec::new();
    let mut guard = 0;
    while !s.done() && guard < guard_limit {
        out.push(s.get());
        s.next();
        guard += 1;
    }
    assert!(guard < guard_limit, "search did not terminate");
    out
}

#[test]
fn default_search_is_already_exhausted() {
    // max < min, so there is nothing to visit.
    assert!(IncreasingDiameterSearch::default().done());
}

#[test]
fn single_value_search_matches_nextpnr() {
    let golden = load_golden();
    let expected = &golden["single 7"];
    assert_eq!(walk(IncreasingDiameterSearch::at(7), 50), *expected);
}

#[test]
fn ranged_search_matches_nextpnr() {
    let golden = load_golden();

    // Starts at, near and past the range edges -- where the sign flipping in
    // `next()` actually decides the order.
    let cases = [
        (5, 0, 10),
        (0, 0, 10),
        (10, 0, 10),
        (2, 0, 10),
        (8, 0, 10),
        (3, 3, 3),
        (0, 0, 1),
        (4, 2, 9),
    ];

    for (start, min, max) in cases {
        let key = format!("range {start} {min} {max}");
        let expected = golden
            .get(&key)
            .unwrap_or_else(|| panic!("golden fixture missing `{key}`"));
        let got = walk(IncreasingDiameterSearch::new(start, min, max), 200);
        assert_eq!(got, *expected, "traversal diverged for {key}");
    }
}

#[test]
fn reset_restarts_the_traversal() {
    let golden = load_golden();
    let expected = &golden["reset 5 0 10"];

    let mut s = IncreasingDiameterSearch::new(5, 0, 10);
    let mut out = Vec::new();
    for _ in 0..4 {
        out.push(s.get());
        s.next();
    }
    s.reset();
    let mut guard = 0;
    while !s.done() && guard < 200 {
        out.push(s.get());
        s.next();
        guard += 1;
    }

    assert_eq!(out, *expected);
}

#[test]
fn traversal_stays_within_range() {
    // The clamp in `get()` is what keeps an off-centre start in bounds.
    for (start, min, max) in [(5, 0, 10), (0, 0, 10), (10, 0, 10), (4, 2, 9)] {
        for v in walk(IncreasingDiameterSearch::new(start, min, max), 200) {
            assert!(
                (min..=max).contains(&v),
                "search left its range: {v} not in {min}..={max}"
            );
        }
    }
}

#[test]
fn traversal_covers_every_value_in_range() {
    // Duplicates are expected, but nothing may be missed -- a location the
    // legaliser never visits is a location it can never legalise into.
    for (start, min, max) in [(5, 0, 10), (0, 0, 10), (10, 0, 10), (4, 2, 9), (0, 0, 1)] {
        let seen = walk(IncreasingDiameterSearch::new(start, min, max), 200);
        for want in min..=max {
            assert!(
                seen.contains(&want),
                "search from {start} over {min}..={max} never visited {want}"
            );
        }
    }
}
