//! Relative pointer types for zero-copy access to binary chip database data.
//!
//! These types are used to represent self-referential pointers within the
//! memory-mapped binary format. The offset field stores a byte offset relative
//! to the address of the offset field itself.

use std::fmt;
use std::marker::PhantomData;

/// A relative pointer to a single value of type `T`.
///
/// The pointer is stored as an `i32` offset relative to the address of the
/// `offset` field itself. This allows the binary format to be loaded at any
/// base address without relocation.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct RelPtr<T> {
    pub offset: i32,
    pub _phantom: PhantomData<T>,
}

impl<T> RelPtr<T> {
    /// Resolve this relative pointer to an absolute pointer.
    ///
    /// Computes `(address_of_self + offset)` to get the absolute address.
    #[inline]
    pub fn get(&self) -> *const T {
        let self_addr = std::ptr::addr_of!(self.offset) as usize;
        let offset = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.offset)) };
        self_addr.wrapping_add(offset as isize as usize) as *const T
    }

    /// Returns true if this pointer has a zero offset (likely null/empty).
    #[inline]
    pub fn is_null(&self) -> bool {
        let offset = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.offset)) };
        offset == 0
    }
}

impl<T> fmt::Debug for RelPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let offset = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.offset)) };
        write!(f, "RelPtr(offset={})", offset)
    }
}

// SAFETY: RelPtr is just an offset value, it does not own anything.
unsafe impl<T: Send> Send for RelPtr<T> {}
unsafe impl<T: Sync> Sync for RelPtr<T> {}

/// A relative pointer to a slice of `T` values.
///
/// Combines a relative offset (to the first element) with a length, providing
/// access to a contiguous array of `T` values in the binary format.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct RelSlice<T> {
    pub offset: i32,
    pub length: u32,
    pub _phantom: PhantomData<T>,
}

impl<T> RelSlice<T> {
    /// Resolve this relative slice to a Rust slice reference.
    ///
    /// # Safety
    /// The caller must ensure the underlying data is valid and properly aligned
    /// for type `T`. Since our POD types are `#[repr(C, packed)]`, alignment
    /// is always 1 byte.
    #[inline]
    pub fn get(&self) -> &[T] {
        let self_addr = std::ptr::addr_of!(self.offset) as usize;
        let offset = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.offset)) };
        let length = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.length)) };
        let ptr = self_addr.wrapping_add(offset as isize as usize) as *const T;
        // SAFETY: The binary format guarantees `length` contiguous elements at the
        // resolved address. All our POD types have alignment 1 (packed).
        unsafe { std::slice::from_raw_parts(ptr, length as usize) }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        let length = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.length)) };
        length as usize
    }

    /// Returns true if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> fmt::Debug for RelSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let offset = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.offset)) };
        let length = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(self.length)) };
        write!(f, "RelSlice(offset={}, length={})", offset, length)
    }
}

// SAFETY: RelSlice is just offset + length values, it does not own anything.
unsafe impl<T: Send> Send for RelSlice<T> {}
unsafe impl<T: Sync> Sync for RelSlice<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn relptr_size() {
        // RelPtr should be exactly 4 bytes (i32 offset + zero-size PhantomData)
        assert_eq!(mem::size_of::<RelPtr<u8>>(), 4);
        assert_eq!(mem::size_of::<RelPtr<u32>>(), 4);
    }

    #[test]
    fn relslice_size() {
        // RelSlice should be exactly 8 bytes (i32 offset + u32 length + zero-size PhantomData)
        assert_eq!(mem::size_of::<RelSlice<u8>>(), 8);
        assert_eq!(mem::size_of::<RelSlice<u32>>(), 8);
    }

    #[test]
    fn relptr_resolve() {
        // Create a buffer: [offset: i32, target_data: u32]
        // offset = 4 (size of i32), pointing right after itself
        #[repr(C, packed)]
        struct TestData {
            ptr: RelPtr<u32>,
            value: u32,
        }
        let data = TestData {
            ptr: RelPtr {
                offset: 4, // points to the next field
                _phantom: PhantomData,
            },
            value: 0xDEADBEEF,
        };
        let resolved = data.ptr.get();
        let val = unsafe { std::ptr::read_unaligned(resolved) };
        assert_eq!(val, 0xDEADBEEF);
    }

    #[test]
    fn relptr_self_reference() {
        // Test that RelPtr with offset 0 points to itself
        let ptr: RelPtr<i32> = RelPtr {
            offset: 0,
            _phantom: PhantomData,
        };
        let resolved = ptr.get();
        // The resolved pointer should point to the offset field itself
        assert_eq!(resolved as usize, std::ptr::addr_of!(ptr.offset) as usize);
    }

    #[test]
    fn relptr_negative_offset() {
        // Create a buffer where the target is before the pointer
        #[repr(C, packed)]
        struct TestData {
            value: u32,
            ptr: RelPtr<u32>,
        }
        let data = TestData {
            value: 42,
            ptr: RelPtr {
                offset: -4, // points back to the previous field
                _phantom: PhantomData,
            },
        };
        let resolved = data.ptr.get();
        let val = unsafe { std::ptr::read_unaligned(resolved) };
        assert_eq!(val, 42);
    }

    #[test]
    fn relslice_resolve() {
        // Layout: [offset: i32, length: u32, data: [u32; 3]]
        #[repr(C, packed)]
        struct TestData {
            slice: RelSlice<u32>,
            values: [u32; 3],
        }
        let data = TestData {
            slice: RelSlice {
                offset: 8, // skip past offset(4) + length(4)
                length: 3,
                _phantom: PhantomData,
            },
            values: [10, 20, 30],
        };
        let resolved = data.slice.get();
        assert_eq!(resolved.len(), 3);
        assert_eq!(resolved[0], 10);
        assert_eq!(resolved[1], 20);
        assert_eq!(resolved[2], 30);
    }

    #[test]
    fn relslice_empty() {
        let slice: RelSlice<u32> = RelSlice {
            offset: 0,
            length: 0,
            _phantom: PhantomData,
        };
        assert!(slice.is_empty());
        assert_eq!(slice.len(), 0);
        assert_eq!(slice.get().len(), 0);
    }

    #[test]
    fn relptr_is_null() {
        let null_ptr: RelPtr<u8> = RelPtr {
            offset: 0,
            _phantom: PhantomData,
        };
        assert!(null_ptr.is_null());

        let non_null_ptr: RelPtr<u8> = RelPtr {
            offset: 42,
            _phantom: PhantomData,
        };
        assert!(!non_null_ptr.is_null());
    }

    #[test]
    fn relptr_debug() {
        let ptr: RelPtr<u8> = RelPtr {
            offset: 123,
            _phantom: PhantomData,
        };
        let debug = format!("{:?}", ptr);
        assert_eq!(debug, "RelPtr(offset=123)");
    }

    #[test]
    fn relslice_debug() {
        let slice: RelSlice<u8> = RelSlice {
            offset: 10,
            length: 5,
            _phantom: PhantomData,
        };
        let debug = format!("{:?}", slice);
        assert_eq!(debug, "RelSlice(offset=10, length=5)");
    }
}
