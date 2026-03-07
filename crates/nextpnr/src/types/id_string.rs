//! Thread-safe string interning via [`IdString`] and [`IdStringPool`].
//!
//! [`IdString`] is a lightweight handle (an i32 index) that represents an
//! interned string. The actual strings are stored in an [`IdStringPool`].

use lasso::{Key, Spur, ThreadedRodeo};
use std::fmt;

// ---------------------------------------------------------------------------
// Backend abstraction
// ---------------------------------------------------------------------------

/// Internal interner backend contract.
///
/// `IdStringPool` depends on this trait, allowing backend replacement without
/// touching call sites across the codebase.
trait InternerBackend {
    fn new() -> Self
    where
        Self: Sized;

    fn intern(&self, s: &str) -> usize;

    fn resolve(&self, key: usize) -> Option<&str>;

    fn len(&self) -> usize;
}

/// Current backend implementation based on `lasso::ThreadedRodeo`.
struct LassoBackend {
    interner: ThreadedRodeo,
}

impl InternerBackend for LassoBackend {
    fn new() -> Self {
        Self {
            interner: ThreadedRodeo::new(),
        }
    }

    fn intern(&self, s: &str) -> usize {
        let spur = self.interner.get_or_intern(s);
        spur.into_usize()
    }

    fn resolve(&self, key: usize) -> Option<&str> {
        let spur = Spur::try_from_usize(key)?;
        self.interner.try_resolve(&spur)
    }

    fn len(&self) -> usize {
        self.interner.len()
    }
}

type Backend = LassoBackend;

/// A lightweight handle to an interned string.
///
/// Index 0 represents the empty string. Valid interned strings have indices >= 1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IdString(pub i32);

impl IdString {
    /// The empty/null IdString (index 0).
    pub const EMPTY: Self = Self(0);

    /// Returns true if this IdString is the empty/null value.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the raw index value.
    #[inline]
    pub const fn index(self) -> i32 {
        self.0
    }
}

impl fmt::Debug for IdString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IdString({})", self.0)
    }
}

/// A thread-safe pool for interning strings.
///
/// Strings are stored once and can be referenced by their [`IdString`] handle.
/// The pool is safe to access from multiple threads.
pub struct IdStringPool {
    backend: Backend,
}

impl IdStringPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            backend: Backend::new(),
        }
    }

    /// Intern a string, returning its [`IdString`] handle.
    ///
    /// If the string has already been interned, returns the existing handle.
    /// If the string is empty, returns [`IdString::EMPTY`].
    pub fn intern(&self, s: &str) -> IdString {
        if s.is_empty() {
            return IdString::EMPTY;
        }

        let key = self.backend.intern(s);
        IdString(key as i32 + 1)
    }

    /// Look up the string corresponding to an [`IdString`] handle.
    ///
    /// Returns `None` if the index is out of range.
    pub fn lookup(&self, id: IdString) -> Option<&str> {
        if id.is_empty() {
            return Some("");
        }

        if id.0 <= 0 {
            return None;
        }

        let key_idx = id.0 as usize - 1;
        self.backend.resolve(key_idx)
    }

    /// Returns the number of interned strings (including the empty string at index 0).
    pub fn len(&self) -> usize {
        self.backend.len() + 1
    }

    /// Returns true if no strings have been interned (only the empty string exists).
    pub fn is_empty(&self) -> bool {
        self.len() <= 1
    }
}

impl Default for IdStringPool {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for IdStringPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.len();
        f.debug_struct("IdStringPool")
            .field("count", &count)
            .finish()
    }
}

/// Trait for types that can be converted to an IdString via a pool.
pub trait IntoIdString {
    fn into_id(self, pool: &IdStringPool) -> IdString;
}

impl IntoIdString for &str {
    fn into_id(self, pool: &IdStringPool) -> IdString {
        pool.intern(self)
    }
}

impl IntoIdString for IdString {
    fn into_id(self, _pool: &IdStringPool) -> IdString {
        self
    }
}

impl IntoIdString for &String {
    fn into_id(self, pool: &IdStringPool) -> IdString {
        pool.intern(self.as_str())
    }
}
