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
