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
