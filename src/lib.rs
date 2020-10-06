//! A fixed-capacity memory buffer allocated on the stack using const generics.
//!
//! This is a low-level utility, useful for implementing higher-level data
//! structures such as fixed-capacity vectors and ring buffers. Since
//! `ConstBuffer`'s main purpose is to build safe abstractions on top of, almost
//! its entire API surface is `unsafe`.
//!
//! `ConstBuffer` does not keep track of which elements are in an initialized
//! state. Furthermore, in order to ensure optimal performance, **no bounds
//! checks are performed** unless debug assertions are enabled. Any misuse of
//! this crate leads to undefined behavior.
//!
//! Building a fixed-capacity vector on top of `ConstBuffer` is pretty
//! straightforward:
//!
//! ```
//! # #![allow(incomplete_features)]
//! # #![feature(const_generics)]
//! # use const_buffer::ConstBuffer;
//! struct ConstVec<T, const N: usize> {
//!     buffer: ConstBuffer<T, N>,
//!     len: usize,
//! }
//!
//! impl<T, const N: usize> ConstVec<T, N> {
//!     fn new() -> Self {
//!         Self { buffer: ConstBuffer::new(), len: 0 }
//!     }
//!
//!     fn push(&mut self, value: T) {
//!         assert!(self.len < N);
//!         unsafe {
//!             self.buffer.write(self.len, value);
//!             self.len += 1;
//!         }
//!     }
//!
//!     fn pop(&mut self) -> Option<T> {
//!         if self.len > 0 {
//!             unsafe {
//!                 self.len -= 1;
//!                 Some(self.buffer.read(self.len))
//!             }
//!         } else {
//!             None
//!         }
//!     }
//!
//!     fn as_slice(&self) -> &[T] {
//!         unsafe { self.buffer.get(..self.len) }
//!     }
//!
//!     fn get(&self, index: usize) -> Option<T> {
//!         if index < self.len { Some(unsafe { self.buffer.read(index) }) } else { None }
//!     }
//! }
//! ```
//!
//! [`RawVec`]: https://github.com/rust-lang/rust/blob/master/src/liballoc/raw_vec.rs

#![no_std]
#![feature(
    associated_type_bounds,
    const_fn_union,
    min_const_generics,
    const_mut_refs,
    maybe_uninit_extra,
    maybe_uninit_ref,
    maybe_uninit_slice,
    untagged_unions,
    unsafe_block_in_unsafe_fn,
    const_ptr_offset
)]
#![allow(incomplete_features)]
#![warn(unsafe_op_in_unsafe_fn)]

#[cfg(test)]
mod tests;

#[cfg(test)]
#[macro_use]
extern crate std;

use core::{
    cmp,
    fmt::{self, Debug, Formatter},
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Bound, Range, RangeBounds},
    ptr,
    slice::SliceIndex,
};

fn to_range(range: impl RangeBounds<usize>, len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n + 1,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    start..end
}

/// A fixed-capacity buffer allocated on the stack using const generics.
pub struct ConstBuffer<T, const N: usize>([MaybeUninit<T>; N]);

impl<T, const N: usize> ConstBuffer<T, N> {
    /// Creates a new `ConstBuffer` from a `MaybeUninit<[T; N]>`.
    #[inline]
    const fn from_maybe_uninit_array(maybe_uninit: MaybeUninit<[T; N]>) -> Self {
        union Transmute<T, const N: usize> {
            maybe_uninit: MaybeUninit<[T; N]>,
            array: ManuallyDrop<[MaybeUninit<T>; N]>,
        }

        // SAFETY: `MaybeUninit<T>` is guaranteed to have the same layout as `T`, and
        // arrays are guaranteed to lay out their elements consecutively, so
        // `MaybeUninit<[T; N]>` and `[MaybeUninit<T>, N]` are guaranteed to have the
        // same layout. See:
        // - https://doc.rust-lang.org/beta/std/mem/union.MaybeUninit.html#layout
        // - https://doc.rust-lang.org/reference/type-layout.html#array-layout
        let array = unsafe { Transmute { maybe_uninit }.array };
        Self(ManuallyDrop::into_inner(array))
    }

    /// Creates a new `ConstBuffer` from an array with the same size.
    #[inline]
    pub const fn from_array(array: [T; N]) -> Self {
        Self::from_maybe_uninit_array(MaybeUninit::new(array))
    }

    /// Creates a new `ConstBuffer` with all elements in an uninitialized state.
    #[inline]
    pub const fn new() -> Self {
        // TODO: use `MaybeUninit::uninit_array` once it is `const`
        Self::from_maybe_uninit_array(MaybeUninit::<[T; N]>::uninit())
    }

    /// Creates a new `ConstBuffer` with all elements in an uninitialized state,
    /// with the memory being filled with `0` bytes. It depends on `T`
    /// whether that already makes for proper initialization. For example,
    /// `ConstBuffer<usize, N>::zeroed()` is initialized, but
    /// `ConstBuffer<&'static i32, N>::zeroed()` is not because references
    /// must not be null.
    ///
    /// # Examples
    ///
    /// Correct usage of this function:
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let buffer = ConstBuffer::<u32, 10>::zeroed();
    /// for i in 0..10 {
    ///     unsafe { assert_eq!(buffer.read(i), 0) };
    /// }
    /// ```
    ///
    /// *Incorrect* usage of this function:
    /// ```no_run
    /// # use const_buffer::ConstBuffer;
    /// let buffer = ConstBuffer::<&'static u32, 10>::zeroed();
    /// let x = unsafe { buffer.read(0) };
    /// ```
    #[inline]
    pub fn zeroed() -> Self {
        Self::from_maybe_uninit_array(MaybeUninit::<[T; N]>::zeroed())
    }

    /// Returns a pointer to the buffer.
    ///
    /// It is up to the caller to ensure that the buffer outlives the pointer
    /// returned from this method, or else it will end up pointing to garbage.
    ///
    /// It is also up to the caller to ensure that the memory this pointer
    /// points to is never written to using this pointer or any pointer derived
    /// from it.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<usize, 6>::zeroed();
    ///
    /// unsafe {
    ///     buffer.write(1, 10);
    ///     buffer.write(3, 30);
    ///     buffer.write(5, 50);
    ///
    ///     let mut p = buffer.as_ptr();
    ///     for i in 0..6 {
    ///         if i % 2 == 1 {
    ///             assert_eq!(std::ptr::read(p), 10 * i);
    ///         }
    ///         p = p.add(1);
    ///     }
    /// }
    /// ```
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.0.as_ptr().cast()
    }

    /// Returns a mutable pointer to the buffer.
    ///
    /// It is up to the caller to ensure that the buffer outlives the pointer
    /// returned from this method, or else it will end up pointing to garbage.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<usize, 6>::zeroed();
    ///
    /// unsafe {
    ///     let mut p = buffer.as_mut_ptr();
    ///     for i in 0..6 {
    ///         if i % 2 == 1 {
    ///             std::ptr::write(p, 10 * i);
    ///         }
    ///         p = p.add(1);
    ///     }
    ///
    ///     assert_eq!(buffer.read(1), 10);
    ///     assert_eq!(buffer.read(3), 30);
    ///     assert_eq!(buffer.read(5), 50);
    /// }
    /// ```
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr().cast()
    }

    /// Reads the element at `index`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `index` is in-bounds, and that the
    /// element at `index` is in an initialized state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(3, 123);
    ///     assert_eq!(buffer.read(3), 123);
    /// }
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn read(&self, index: usize) -> T {
        debug_assert!(index < N);

        // SAFETY:
        // - `get_unchecked` requires that `index` is in-bounds
        // - `assume_init_read` requires that the element at `index` is in an
        //   initialized state
        // Both of these requirements are required by `read` as well.
        unsafe { self.0.get_unchecked(index).assume_init_read() }
    }

    /// Sets the element at `index`.
    ///
    /// This overwrites any previous element at that index without dropping it,
    /// and returns a mutable reference to the (now safely initialized) element
    /// at `index`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `index` is in-bounds. The old
    /// contents at `index` aren't dropped, so the element at `index` does not
    /// need to be in an initialized state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     let x = buffer.write(3, 123);
    ///     *x *= 3;
    ///     assert_eq!(buffer.read(3), 369);
    /// }
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn write(&mut self, index: usize, value: T) -> &mut T {
        debug_assert!(index < N);

        // SAFETY: `get_unchecked_mut` requires that `index` is in-bounds, which `write`
        // requires as well.
        unsafe { self.0.get_unchecked_mut(index) }.write(value)
    }

    /// Returns a reference to an element or subslice depending on the type of
    /// index.
    ///
    /// - Given a position, this returns a reference to the element at that
    ///   position.
    /// - Given a range, this returns the subslice corresponding to that range.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the position or range is
    /// in-bounds, and that the corresponding elements are in an initialized
    /// state.
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(3, 30);
    ///     buffer.write(4, 40);
    ///     buffer.write(5, 50);
    ///
    ///     assert_eq!(buffer.get(3), &30);
    ///     assert_eq!(buffer.get(4..6), &[40, 50]);
    /// }
    /// ```
    ///
    /// *Incorrect* usage of this method:
    /// ```no_run
    /// # use const_buffer::ConstBuffer;
    /// let buffer = ConstBuffer::<u32, 10>::new();
    /// let x = unsafe { buffer.get(0) };
    /// // We have created a reference to an uninitialized value! This is
    /// // undefined behavior.
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn get<'a, I>(&'a self, index: I) -> &I::Output
    where
        I: BufferIndex<'a, T>,
    {
        // SAFETY: `BufferIndex::get` has the same safety requirements as this method.
        unsafe { index.get(self) }
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the position or range is
    /// in-bounds, and that the corresponding elements are in an initialized
    /// state.
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(3, 30);
    ///     buffer.write(4, 40);
    ///     buffer.write(5, 50);
    ///
    ///     *buffer.get_mut(3) += 5;
    ///     buffer.get_mut(4..6).reverse();
    ///
    ///     assert_eq!(buffer.read(3), 35);
    ///     assert_eq!(buffer.read(4), 50);
    ///     assert_eq!(buffer.read(5), 40);
    /// }
    /// ```
    ///
    /// *Incorrect* usage of this method:
    /// ```no_run
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     // We create a reference to uninitialized memory which is
    ///     // undefined behavior, despite not reading from it.
    ///     *buffer.get_mut(3) = 5;
    ///     assert_eq!(buffer.read(3), 5);
    /// }
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn get_mut<'a, I>(&'a mut self, index: I) -> &mut I::Output
    where
        I: BufferIndex<'a, T>,
    {
        // SAFETY: `BufferIndex::get_mut` has the same safety requirements as this
        // method.
        unsafe { index.get_mut(self) }
    }

    /// Swaps the elements at indices `i` and `j`. `i` and `j` may be equal. The
    /// elements at the given indices are not required to be in an initialized
    /// state.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `i` and `j` are in-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(3, 10);
    ///     buffer.write(5, 20);
    ///
    ///     buffer.swap(3, 3);
    ///     buffer.swap(3, 5);
    ///
    ///     assert_eq!(buffer.read(3), 20);
    ///     assert_eq!(buffer.read(5), 10);
    /// }
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn swap(&mut self, i: usize, j: usize) {
        debug_assert!(i < N && j < N);

        // SAFETY: `i` and `j` are required to be in-bounds.
        unsafe {
            let x = self.0.as_mut_ptr().add(i);
            let y = self.0.as_mut_ptr().add(j);
            ptr::swap(x, y);
        }
    }

    /// Swaps the elements at indices `i` and `j`. `i` and `j` must not be
    /// equal to each other. The elements at the given indices are not required
    /// to be in an initialized state.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `i` and `j` are in-bounds, and
    /// that `i` and `j` are not equal to each other.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(3, 10);
    ///     buffer.write(5, 20);
    ///
    ///     buffer.swap(3, 5);
    ///
    ///     assert_eq!(buffer.read(3), 20);
    ///     assert_eq!(buffer.read(5), 10);
    /// }
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn swap_nonoverlapping(&mut self, i: usize, j: usize) {
        debug_assert!(i < N && j < N && i != j);

        // SAFETY: `i` and `j` are required to be in-bounds and distinct, which implies
        // that the bytes being swapped do not overlap.
        unsafe {
            let x = self.0.as_mut_ptr().add(i);
            let y = self.0.as_mut_ptr().add(j);
            ptr::swap_nonoverlapping(x, y, 1);
        }
    }

    /// Creates a new buffer with a potentially different size.
    ///
    /// If the new buffer is larger than the original buffer, all the contents
    /// of the original buffer are copied over to the new buffer, and the
    /// rest will be uninitialized.
    ///
    /// If the new buffer is smaller than the original buffer, the new buffer
    /// will be filled entirely with contents of the original buffer, ignoring
    /// any excess elements at the end.
    ///
    /// Note that this simply copies the bytes of the original buffer to the
    /// new buffer, and it does not call `clone` on any of the elements in the
    /// buffer. Therefore, if you end up reading the same element from both
    /// buffers, it is your responsibility to ensure that that data may indeed
    /// be duplicated.
    ///
    /// If you want `clone` to be called on the elements in the buffer, consider
    /// using [`clone_from_slice`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(3, 30);
    ///     buffer.write(7, 70);
    /// }
    ///
    /// let small: ConstBuffer<u32, 5> = buffer.resize();
    /// let large: ConstBuffer<u32, 15> = buffer.resize();
    ///
    /// unsafe {
    ///     assert_eq!(small.read(3), 30);
    ///
    ///     // This read would be out-of-bounds.
    ///     // assert_eq!(small.read(7), 70);
    ///
    ///     assert_eq!(large.read(3), 30);
    ///     assert_eq!(large.read(7), 70);
    /// }
    /// ```
    ///
    /// [`clone_from_slice`]: ConstBuffer::clone_from_slice
    #[inline]
    pub fn resize<const M: usize>(&self) -> ConstBuffer<T, M> {
        let mut new = ConstBuffer::new();

        let src = self.0.as_ptr();
        let dest = new.0.as_mut_ptr();
        let len = cmp::min(N, M);

        // SAFETY: `len` is in-bounds for both `self` and `new` by construction, and the
        // memory regions don't overlap because they're part of different
        // `ConstBuffer`s.
        unsafe { ptr::copy_nonoverlapping(src, dest, len) };

        new
    }

    /// Copies elements from one part of the buffer to another part of itself.
    ///
    /// `src` is the range within `self` to copy from. This range is allowed to
    /// contain uninitialized elements. `dest` is the starting index of the
    /// range within `self` to copy to, which will have the same length as
    /// `src`. The two ranges may overlap.
    ///
    /// Note that unlike [`slice::copy_within`], this method does **not**
    /// require that `T` implements [`Copy`].
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the two ranges are in-bounds.
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(2, 10);
    ///     buffer.write(5, 20);
    ///     buffer.write(7, 30);
    ///
    ///     // This overwrites the elements in 1..7 with the elements in
    ///     // 4..10, so 20 ends up at index 2 and 30 at index 4.
    ///     buffer.copy_within(4.., 1);
    ///
    ///     assert_eq!(buffer.read(2), 20);
    ///     assert_eq!(buffer.read(4), 30);
    ///
    ///     // The element at index 7 is still there:
    ///     assert_eq!(buffer.read(7), 30);
    /// }
    /// ```
    ///
    /// *Incorrect* usage of this method:
    /// ```no_run
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// // This will try to copy the elements from 0..5 to 8..13,
    /// // which is out-of-bounds.
    /// unsafe { buffer.copy_within(..5, 8); }
    /// ```
    ///
    /// [`slice::copy_within`]: https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.copy_within
    /// [`Copy`]: core::marker::Copy
    #[inline]
    #[track_caller]
    pub unsafe fn copy_within<R>(&mut self, src: R, dest: usize)
    where
        R: RangeBounds<usize>,
    {
        // we can't call `copy_within` on `self.0` because `MaybeUninit` isn't `Copy`

        let src = to_range(src, N);
        debug_assert!(src.start <= src.end && src.end <= N && dest + src.len() <= N);

        // SAFETY: The two ranges are required to be in-bounds.
        unsafe {
            let src_ptr = self.0.as_ptr().add(src.start);
            let dest_ptr = self.0.as_mut_ptr().add(dest);
            ptr::copy(src_ptr, dest_ptr, src.len());
        }
    }

    /// Copies elements from one part of the buffer to another part of itself.
    /// The source and destination must _not_ overlap.
    ///
    /// `src` is the range within `self` to copy from. This range is allowed to
    /// contain uninitialized elements. `dest` is the starting index of the
    /// range within `self` to copy to, which will have the same length as
    /// `src`. The two ranges must not overlap.
    ///
    /// Note that unlike [`slice::copy_within`], this method does **not**
    /// require that `T` implements [`Copy`].
    ///
    /// For ranges that might overlap, use [`copy_within`] instead.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the two ranges are in-bounds, and
    /// that the two ranges don't overlap.
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(5, 10);
    ///     buffer.write(7, 20);
    ///     buffer.write(9, 30);
    ///
    ///     buffer.copy_within_nonoverlapping(6.., 2);
    ///
    ///     assert_eq!(buffer.read(3), 20);
    ///     assert_eq!(buffer.read(5), 30);
    ///     assert_eq!(buffer.read(7), 20);
    ///     assert_eq!(buffer.read(9), 30);
    /// }
    /// ```
    ///
    /// *Incorrect* usage of this method:
    /// ```no_run
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// // This will try to copy the elements from 4..8 to 2..6,
    /// // which are overlapping ranges.
    /// unsafe { buffer.copy_within(4..8, 2); }
    /// ```
    ///
    /// [`slice::copy_within`]: https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.copy_within
    /// [`Copy`]: core::marker::Copy
    /// [`copy_within`]: ConstBuffer::copy_within
    #[inline]
    #[track_caller]
    pub unsafe fn copy_within_nonoverlapping<R>(&mut self, src: R, dest: usize)
    where
        R: RangeBounds<usize>,
    {
        let src = to_range(src, N);
        debug_assert!(src.start <= src.end && src.end <= N && dest + src.len() <= N);
        debug_assert!(
            src.len() <= cmp::max(src.start, dest) - cmp::min(src.start, dest),
            "attempt to copy to overlapping memory"
        );

        // SAFETY: The two ranges are required to be in-bounds and to not overlap.
        unsafe {
            let src_ptr = self.as_ptr().add(src.start);
            let dest_ptr = self.as_mut_ptr().add(dest);
            ptr::copy_nonoverlapping(src_ptr, dest_ptr, src.len());
        }
    }

    /// Copies the elements from the given slice into `self`, starting at
    /// position `index`.
    ///
    /// Note that unlike [`slice::copy_from_slice`], this method does **not**
    /// require that `T` implements [`Copy`]. It is your responsibility to make
    /// sure that this data can safely be duplicated. If not, consider using
    /// [`clone_from_slice`] instead.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the range the slice is copied to
    /// is in-bounds.
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.copy_from_slice(3, &[1, 4]);
    ///     assert_eq!(buffer.read(3), 1);
    ///     assert_eq!(buffer.read(4), 4);
    /// }
    /// ```
    ///
    /// *Incorrect* usage of this method:
    /// ```no_run
    /// # use const_buffer::ConstBuffer;
    /// let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let mut buffer = ConstBuffer::<Vec<u32>, 10>::new();
    ///
    /// unsafe {
    ///     buffer.copy_from_slice(3, &vec);
    ///     let x = buffer.read(4);
    ///     // The drop handler of `x` is executed here, which
    ///     // frees the vector's memory and has `vec[1]` pointing
    ///     // to garbage memory as a result.
    /// }
    /// ```
    ///
    /// [`slice::copy_from_slice`]: https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.copy_from_slice
    /// [`Copy`]: core::marker::Copy
    /// [`clone_from_slice`]: ConstBuffer::clone_from_slice
    #[inline]
    #[track_caller]
    pub unsafe fn copy_from_slice(&mut self, index: usize, slice: &[T]) {
        debug_assert!(index + slice.len() <= N);

        // SAFETY: `index` is required to be in-bounds.
        let dest = unsafe { self.as_mut_ptr().add(index) };

        // SAFETY: The range `index..(index + slice.len())` is required to be in-bounds,
        // and `slice` is guaranteed to not overlap with `self` because this method
        // requires a unique reference to `self`.
        unsafe { slice.as_ptr().copy_to_nonoverlapping(dest, slice.len()) };
    }

    /// Clones the elements from the given slice into `self`, starting at
    /// position `index`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the range the slice is copied to
    /// is in-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let mut buffer = ConstBuffer::<Vec<u32>, 10>::new();
    ///
    /// unsafe {
    ///     buffer.clone_from_slice(3, &vec);
    ///     let mut x = buffer.read(4);
    ///
    ///     // `x` is a clone of `vec[1]`, so this will not affect the
    ///     // original vector.
    ///     x.reverse();
    ///     assert_eq!(x, &[6, 5, 4]);
    /// }
    ///
    /// assert_eq!(vec[1], &[4, 5, 6]);
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn clone_from_slice(&mut self, index: usize, slice: &[T])
    where
        T: Clone,
    {
        debug_assert!(index + slice.len() <= N);
        (index..).zip(slice).for_each(|(i, x)| {
            // SAFETY: It is up to the caller of `clone_from_slice` to ensure that
            // `index + slice.len() <= N`. It follows that `i < N`, which is exactly what
            // `write` requires.
            unsafe { self.write(i, x.clone()) };
        });
    }

    /// Returns a [`MaybeUninit`] slice to the buffer.
    /// Useful when you want to read `MaybeUninit<T>` values that may or may
    /// not be in an initialized state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     buffer.write(4, 1);
    ///     buffer.write(6, 3);
    ///
    ///     let slice = &buffer.as_maybe_uninit_slice()[3..];
    ///     assert_eq!(slice[1].assume_init(), 1);
    ///     assert_eq!(slice[3].assume_init(), 3);
    /// }
    /// ```
    ///
    /// [`MaybeUninit`]: core::mem::MaybeUninit
    #[inline]
    pub fn as_maybe_uninit_slice(&self) -> &[MaybeUninit<T>] {
        &self.0
    }

    /// Returns a mutable [`MaybeUninit`] slice to the
    /// buffer. Useful when you want to write `MaybeUninit<T>` values that
    /// may or may not be in an initialized state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use const_buffer::ConstBuffer;
    /// use std::mem::MaybeUninit;
    ///
    /// let mut buffer = ConstBuffer::<u32, 10>::new();
    ///
    /// unsafe {
    ///     let slice = &mut buffer.as_maybe_uninit_mut_slice()[3..];
    ///     slice[1] = MaybeUninit::new(1);
    ///     slice[3] = MaybeUninit::new(3);
    ///
    ///     assert_eq!(buffer.read(4), 1);
    ///     assert_eq!(buffer.read(6), 3);
    /// }
    /// ```
    ///
    /// [`MaybeUninit`]: core::mem::MaybeUninit
    #[inline]
    pub fn as_maybe_uninit_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.0
    }
}

impl<T, const N: usize> Default for ConstBuffer<T, N> {
    /// Creates an empty buffer.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Clone for ConstBuffer<T, N> {
    /// Returns a copy of the buffer.
    ///
    /// Note that this simply copies the bytes of the original buffer to the
    /// new buffer, and it does not call `clone` on any of the elements in the
    /// buffer. Therefore, if you end up reading the same element from both
    /// buffers, it is your responsibility to ensure that that data may indeed
    /// be duplicated.
    ///
    /// If you want `clone` to be called on the elements in the buffer, consider
    /// using [`clone_from_slice`] instead.
    ///
    /// [`clone_from_slice`]: ConstBuffer::clone_from_slice
    #[inline]
    fn clone(&self) -> Self {
        self.resize()
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        let src = source.0.as_ptr();
        let dest = self.0.as_mut_ptr();

        // SAFETY: The memory regions don't overlap because they correspond to different
        // `ConstBuffer`s.
        unsafe { ptr::copy_nonoverlapping(src, dest, N) };
    }
}

impl<T, const N: usize> Debug for ConstBuffer<T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.pad(core::any::type_name::<Self>())
    }
}

impl<T, const N: usize> From<[T; N]> for ConstBuffer<T, N> {
    #[inline]
    fn from(array: [T; N]) -> Self {
        Self::from_array(array)
    }
}

impl<T, const N: usize> From<[MaybeUninit<T>; N]> for ConstBuffer<T, N> {
    #[inline]
    fn from(array: [MaybeUninit<T>; N]) -> Self {
        Self(array)
    }
}

impl<T, const N: usize> From<MaybeUninit<[T; N]>> for ConstBuffer<T, N> {
    #[inline]
    fn from(maybe_uninit: MaybeUninit<[T; N]>) -> Self {
        Self::from_maybe_uninit_array(maybe_uninit)
    }
}

/// A helper trait that generalizes over positions and ranges to be used as a
/// trait bound for [`get`] and [`get_mut`].
///
/// This trait is automatically implemented for types implementing
/// [`SliceIndex`], and it is not useful to implement this trait yourself.
///
/// [`get`]: ConstBuffer::get
/// [`get_mut`]: ConstBuffer::get_mut
/// [`SliceIndex`]: core::slice::SliceIndex
pub trait BufferIndex<'a, T> {
    type Output: ?Sized;

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the index is in-bounds, and that
    /// the corresponding output is in an initialized state.
    unsafe fn get<const N: usize>(self, buffer: &'a ConstBuffer<T, N>) -> &'a Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the index is in-bounds, and that
    /// the corresponding output is in an initialized state.
    unsafe fn get_mut<const N: usize>(
        self,
        buffer: &'a mut ConstBuffer<T, N>,
    ) -> &'a mut Self::Output;
}

impl<'a, T, O: ?Sized, I> BufferIndex<'a, T> for I
where
    I: SliceIndex<[MaybeUninit<T>], Output: UninitWrapper<Output = O> + 'a> + Clone,
{
    type Output = O;

    unsafe fn get<const N: usize>(self, buffer: &'a ConstBuffer<T, N>) -> &'a Self::Output {
        debug_assert!(buffer.0.get(self.clone()).is_some());

        // SAFETY: The index is required to be in-bounds.
        let x = unsafe { buffer.0.get_unchecked(self) };

        // SAFETY: The output at the index is required to be in an initialized state.
        unsafe { x.get() }
    }

    unsafe fn get_mut<const N: usize>(
        self,
        buffer: &'a mut ConstBuffer<T, N>,
    ) -> &'a mut Self::Output {
        debug_assert!(buffer.0.get(self.clone()).is_some());

        // SAFETY: The index is required to be in-bounds.
        let x = unsafe { buffer.0.get_unchecked_mut(self) };

        // SAFETY: The output at the index is required to be in an initialized state.
        unsafe { x.get_mut() }
    }
}

/// A helper trait that generalizes over [`MaybeUninit<T>`] and
/// `[MaybeUninit<T>]` for the purpose of making [`get`] and [`get_mut`] work
/// with both positions and ranges.
///
/// It is not useful to implement this trait yourself.
///
/// [`MaybeUninit<T>`]: core::mem::MaybeUninit
/// [`get`]: ConstBuffer::get
/// [`get_mut`]: ConstBuffer::get_mut
pub trait UninitWrapper {
    type Output: ?Sized;

    /// Returns a shared reference to the contained value.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the content is fully initialized.
    unsafe fn get(&self) -> &Self::Output;

    /// Returns a mutable reference to the contained value.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the content is fully initialized.
    unsafe fn get_mut(&mut self) -> &mut Self::Output;
}

impl<T> UninitWrapper for MaybeUninit<T> {
    type Output = T;

    unsafe fn get(&self) -> &Self::Output {
        // SAFETY: The content is required to be fully initialized.
        unsafe { self.assume_init_ref() }
    }

    unsafe fn get_mut(&mut self) -> &mut Self::Output {
        // SAFETY: The content is required to be fully initialized.
        unsafe { self.assume_init_mut() }
    }
}

impl<T> UninitWrapper for [MaybeUninit<T>] {
    type Output = [T];

    unsafe fn get(&self) -> &Self::Output {
        // SAFETY: The content is required to be fully initialized.
        unsafe { MaybeUninit::slice_assume_init_ref(self) }
    }

    unsafe fn get_mut(&mut self) -> &mut Self::Output {
        // SAFETY: The content is required to be fully initialized.
        unsafe { MaybeUninit::slice_assume_init_mut(self) }
    }
}
