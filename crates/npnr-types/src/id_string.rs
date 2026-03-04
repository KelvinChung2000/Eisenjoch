//! Thread-safe string interning via [`IdString`] and [`IdStringPool`].
//!
//! [`IdString`] is a lightweight handle (an i32 index) that represents an
//! interned string. The actual strings are stored in an [`IdStringPool`] which
//! uses a `RwLock`-protected hash map for deduplication and a `Vec` for
//! index-to-string lookup.

use rustc_hash::FxHashMap;
use std::fmt;
use std::sync::RwLock;

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
/// The pool is safe to access from multiple threads via `RwLock`.
pub struct IdStringPool {
    str_to_idx: RwLock<FxHashMap<String, i32>>,
    idx_to_str: RwLock<Vec<String>>,
}

impl IdStringPool {
    /// Create a new empty pool.
    ///
    /// Index 0 is reserved for the empty string.
    pub fn new() -> Self {
        let pool = Self {
            str_to_idx: RwLock::new(FxHashMap::default()),
            idx_to_str: RwLock::new(vec![String::new()]), // index 0 = empty
        };
        pool
    }

    /// Intern a string, returning its [`IdString`] handle.
    ///
    /// If the string has already been interned, returns the existing handle.
    /// If the string is empty, returns [`IdString::EMPTY`].
    pub fn intern(&self, s: &str) -> IdString {
        if s.is_empty() {
            return IdString::EMPTY;
        }

        // Fast path: check if already interned (read lock only).
        {
            let map = self.str_to_idx.read().unwrap();
            if let Some(&idx) = map.get(s) {
                return IdString(idx);
            }
        }

        // Slow path: acquire write lock and insert.
        let mut map = self.str_to_idx.write().unwrap();
        // Double-check after acquiring write lock (another thread may have inserted).
        if let Some(&idx) = map.get(s) {
            return IdString(idx);
        }

        let mut strings = self.idx_to_str.write().unwrap();
        let idx = strings.len() as i32;
        strings.push(s.to_owned());
        map.insert(s.to_owned(), idx);

        IdString(idx)
    }

    /// Look up the string corresponding to an [`IdString`] handle.
    ///
    /// Returns `None` if the index is out of range.
    pub fn lookup(&self, id: IdString) -> Option<String> {
        let strings = self.idx_to_str.read().unwrap();
        let idx = id.0 as usize;
        strings.get(idx).cloned()
    }

    /// Returns the number of interned strings (including the empty string at index 0).
    pub fn len(&self) -> usize {
        self.idx_to_str.read().unwrap().len()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        assert_eq!(IdString::EMPTY.index(), 0);
        assert!(IdString::EMPTY.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let id = IdString::default();
        assert_eq!(id, IdString::EMPTY);
        assert!(id.is_empty());
    }

    #[test]
    fn intern_returns_same_id_for_same_string() {
        let pool = IdStringPool::new();
        let a = pool.intern("hello");
        let b = pool.intern("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn intern_returns_different_ids_for_different_strings() {
        let pool = IdStringPool::new();
        let a = pool.intern("hello");
        let b = pool.intern("world");
        assert_ne!(a, b);
    }

    #[test]
    fn intern_empty_string_returns_empty() {
        let pool = IdStringPool::new();
        let id = pool.intern("");
        assert_eq!(id, IdString::EMPTY);
    }

    #[test]
    fn lookup_interned_string() {
        let pool = IdStringPool::new();
        let id = pool.intern("test");
        assert_eq!(pool.lookup(id).as_deref(), Some("test"));
    }

    #[test]
    fn lookup_empty_id() {
        let pool = IdStringPool::new();
        assert_eq!(pool.lookup(IdString::EMPTY).as_deref(), Some(""));
    }

    #[test]
    fn lookup_invalid_id() {
        let pool = IdStringPool::new();
        assert_eq!(pool.lookup(IdString(999)), None);
    }

    #[test]
    fn pool_len() {
        let pool = IdStringPool::new();
        assert_eq!(pool.len(), 1); // empty string at index 0
        pool.intern("a");
        assert_eq!(pool.len(), 2);
        pool.intern("b");
        assert_eq!(pool.len(), 3);
        pool.intern("a"); // duplicate, no growth
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn pool_is_empty() {
        let pool = IdStringPool::new();
        assert!(pool.is_empty());
        pool.intern("x");
        assert!(!pool.is_empty());
    }

    #[test]
    fn ids_are_sequential() {
        let pool = IdStringPool::new();
        let a = pool.intern("first");
        let b = pool.intern("second");
        let c = pool.intern("third");
        assert_eq!(a.index(), 1);
        assert_eq!(b.index(), 2);
        assert_eq!(c.index(), 3);
    }

    #[test]
    fn id_string_copy_semantics() {
        let id = IdString(42);
        let copy = id;
        assert_eq!(id, copy);
    }

    #[test]
    fn id_string_hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(IdString(1));
        set.insert(IdString(2));
        set.insert(IdString(1));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn id_string_debug() {
        let id = IdString(42);
        assert_eq!(format!("{:?}", id), "IdString(42)");
    }

    #[test]
    fn pool_debug() {
        let pool = IdStringPool::new();
        let debug = format!("{:?}", pool);
        assert!(debug.contains("IdStringPool"));
        assert!(debug.contains("count"));
    }

    #[test]
    fn thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(IdStringPool::new());
        let mut handles = vec![];

        for i in 0..10 {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let s = format!("string_{}", i);
                let id = pool.intern(&s);
                assert!(!id.is_empty());
                assert_eq!(pool.lookup(id).as_deref(), Some(s.as_str()));
                id
            }));
        }

        let ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All IDs should be distinct
        let mut unique = std::collections::HashSet::new();
        for id in &ids {
            unique.insert(*id);
        }
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn concurrent_duplicate_inserts() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(IdStringPool::new());
        let mut handles = vec![];

        // All threads intern the same string
        for _ in 0..10 {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || pool.intern("shared")));
        }

        let ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All should get the same ID
        for id in &ids {
            assert_eq!(*id, ids[0]);
        }

        // Pool should only have 2 entries (empty + "shared")
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn non_empty_id_is_not_empty() {
        let pool = IdStringPool::new();
        let id = pool.intern("notempty");
        assert!(!id.is_empty());
    }
}
