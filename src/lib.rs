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
    const_fn,
    const_fn_union,
    const_generics,
    const_mut_refs,
    maybe_uninit_extra,
    maybe_uninit_ref,
    maybe_uninit_slice_assume_init,
    track_caller,
    untagged_unions
)]
#![allow(incomplete_features)]

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
    slice::{self, SliceIndex},
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
        // TODO: use `slice::as_mut_ptr` once it is `const`
        (&mut self.0 as *mut [MaybeUninit<T>; N]).cast()
    }

    /// Reads the element at `index`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `index` is not out of bounds, and
    /// that the element at `index` is in an initialized state.
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
        self.0.get_unchecked(index).read()
    }

    /// Sets the element at `index`.
    ///
    /// This overwrites any previous element at that index without dropping it,
    /// and returns a mutable reference to the (now safely initialized) element
    /// at `index`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `index` is not out of bounds. The
    /// old contents at `index` aren't dropped, so the element at `index` does
    /// not need to be in an initialized state.
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
        self.0.get_unchecked_mut(index).write(value)
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
    /// It is up to the caller to ensure that the position or range is not out
    /// of bounds, and that the corresponding elements are in an initialized
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
    where I: BufferIndex<'a, T> {
        index.get(self)
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the position or range is not out
    /// of bounds, and that the corresponding elements are in an initialized
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
    where I: BufferIndex<'a, T> {
        index.get_mut(self)
    }

    /// Swaps the elements at indices `i` and `j`. `i` and `j` may be equal.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `i` and `j`  are not out of
    /// bounds.
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
        ptr::swap(self.as_mut_ptr().add(i), self.as_mut_ptr().add(j));
    }

    /// Swaps the elements at indices `i` and `j`. `i` and `j` must not be
    /// equal to each other.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that `i` and `j`  are not out of
    /// bounds, and that `i` and `j` are not equal to each other.
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
        ptr::swap_nonoverlapping(self.as_mut_ptr().add(i), self.as_mut_ptr().add(j), 1);
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
        unsafe { self.0.as_ptr().copy_to_nonoverlapping(new.0.as_mut_ptr(), cmp::min(N, M)) };
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
    /// It is up to the caller to ensure that the two ranges are in-bounds, and
    /// that the end of `src` is before the start.
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
    where R: RangeBounds<usize> {
        let src = to_range(src, N);
        debug_assert!(src.start <= src.end && src.end <= N && dest + src.len() <= N);
        // we can't call `copy_within` on `self.0` because `MaybeUninit` isn't `Copy`
        self.0.as_ptr().add(src.start).copy_to(self.0.as_mut_ptr().add(dest), src.len());
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
    /// It is up to the caller to ensure that the two ranges are in-bounds, that
    /// the two ranges don't overlap, and that the end of `src` is before
    /// the start.
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
    where R: RangeBounds<usize> {
        let src = to_range(src, N);
        debug_assert!(src.start <= src.end && src.end <= N && dest + src.len() <= N);
        debug_assert!(
            src.len() <= cmp::max(src.start, dest) - cmp::min(src.start, dest),
            "attempt to copy to overlapping memory"
        );
        self.as_ptr().add(src.start).copy_to_nonoverlapping(self.as_mut_ptr().add(dest), src.len())
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
        slice.as_ptr().copy_to_nonoverlapping(self.as_mut_ptr().add(index), slice.len());
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
    ///     // `x` is a clone of `vec[1]`, so this will no affect the
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
    where T: Clone {
        debug_assert!(index + slice.len() <= N);
        (index..).zip(slice).for_each(|(i, x)| {
            self.write(i, x.clone());
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
        unsafe { slice::from_raw_parts(self.0.as_ptr(), N) }
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
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr(), N) }
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
        unsafe { source.0.as_ptr().copy_to_nonoverlapping(self.0.as_mut_ptr(), N) };
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

    unsafe fn get<const N: usize>(self, buffer: &'a ConstBuffer<T, N>) -> &'a Self::Output;
    unsafe fn get_mut<const N: usize>(
        self,
        buffer: &'a mut ConstBuffer<T, N>,
    ) -> &'a mut Self::Output;
}

impl<'a, T, O: ?Sized, I> BufferIndex<'a, T> for I
where I: SliceIndex<[MaybeUninit<T>], Output: UninitWrapper<Output = O> + 'a> + Clone
{
    type Output = O;

    unsafe fn get<const N: usize>(self, buffer: &'a ConstBuffer<T, N>) -> &'a Self::Output {
        debug_assert!(buffer.0.get(self.clone()).is_some());
        buffer.0.get_unchecked(self).get()
    }

    unsafe fn get_mut<const N: usize>(
        self,
        buffer: &'a mut ConstBuffer<T, N>,
    ) -> &'a mut Self::Output
    {
        debug_assert!(buffer.0.get(self.clone()).is_some());
        buffer.0.get_unchecked_mut(self).get_mut()
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

    unsafe fn get(&self) -> &Self::Output;
    unsafe fn get_mut(&mut self) -> &mut Self::Output;
}

impl<T> UninitWrapper for MaybeUninit<T> {
    type Output = T;

    unsafe fn get(&self) -> &Self::Output {
        self.get_ref()
    }

    unsafe fn get_mut(&mut self) -> &mut Self::Output {
        self.get_mut()
    }
}

impl<T> UninitWrapper for [MaybeUninit<T>] {
    type Output = [T];

    unsafe fn get(&self) -> &Self::Output {
        MaybeUninit::slice_get_ref(self)
    }

    unsafe fn get_mut(&mut self) -> &mut Self::Output {
        MaybeUninit::slice_get_mut(self)
    }
}
