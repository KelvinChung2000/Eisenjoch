//! Deterministic RNG for reproducible place-and-route results.
//!
//! Uses a simple xorshift64 algorithm to produce a deterministic sequence of
//! pseudo-random numbers from a given seed. This ensures that repeated runs
//! with the same seed produce identical placements and routes.

/// A simple xorshift64-based deterministic RNG.
///
/// Designed for reproducibility rather than cryptographic quality.
/// The same seed always produces the same sequence of outputs.
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Create a new RNG with the given seed.
    ///
    /// A seed of 0 is adjusted to 1 to avoid the xorshift zero fixpoint.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    /// Generate the next 64-bit pseudo-random value.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate the next 32-bit pseudo-random value.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Generate a random value in the range `[0, max)`.
    ///
    /// # Panics
    ///
    /// Panics if `max` is 0.
    #[inline]
    pub fn next_range(&mut self, max: u32) -> u32 {
        self.next_u32() % max
    }

    /// Shuffle a slice in place using the Fisher-Yates algorithm.
    ///
    /// Produces a uniformly random permutation given the current RNG state.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.next_range((i + 1) as u32) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_seed_is_adjusted() {
        let mut rng = DeterministicRng::new(0);
        // Should not get stuck at 0
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    #[test]
    fn deterministic_output() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let mut a = DeterministicRng::new(1);
        let mut b = DeterministicRng::new(2);
        // Very unlikely to be equal unless the RNG is broken
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn next_u32_truncates() {
        let mut rng = DeterministicRng::new(123);
        let v64 = {
            let mut r2 = DeterministicRng::new(123);
            r2.next_u64()
        };
        let v32 = rng.next_u32();
        assert_eq!(v32, v64 as u32);
    }

    #[test]
    fn next_range_bounded() {
        let mut rng = DeterministicRng::new(99);
        for _ in 0..1000 {
            let v = rng.next_range(10);
            assert!(v < 10);
        }
    }

    #[test]
    #[should_panic]
    fn next_range_zero_panics() {
        let mut rng = DeterministicRng::new(1);
        rng.next_range(0);
    }

    #[test]
    fn shuffle_empty() {
        let mut rng = DeterministicRng::new(1);
        let mut data: Vec<i32> = vec![];
        rng.shuffle(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn shuffle_single() {
        let mut rng = DeterministicRng::new(1);
        let mut data = vec![42];
        rng.shuffle(&mut data);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = DeterministicRng::new(1);
        let mut data: Vec<i32> = (0..20).collect();
        rng.shuffle(&mut data);
        data.sort();
        let expected: Vec<i32> = (0..20).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn shuffle_deterministic() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);
        let mut data1: Vec<i32> = (0..50).collect();
        let mut data2: Vec<i32> = (0..50).collect();
        rng1.shuffle(&mut data1);
        rng2.shuffle(&mut data2);
        assert_eq!(data1, data2);
    }

    #[test]
    fn shuffle_actually_shuffles() {
        let mut rng = DeterministicRng::new(12345);
        let original: Vec<i32> = (0..20).collect();
        let mut data = original.clone();
        rng.shuffle(&mut data);
        // Extremely unlikely to get the identity permutation with 20 elements.
        assert_ne!(data, original);
    }
}
