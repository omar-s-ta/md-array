//! A dense, fixed-rank multi-dimensional array.
//!
//! [`MdArray<T, N>`] stores `N`-dimensional data in a single contiguous
//! [`Vec<T>`] using **row-major** (C) order: the last index varies fastest, so
//! the element at `[i0, i1, …, i_{N-1}]` lives at flat offset
//! `((i0 * d1 + i1) * d2 + i2) … + i_{N-1}`, where `dk` is the length of
//! dimension `k`. The rank `N` is a const generic, so it is fixed at compile
//! time while the per-dimension lengths are chosen at runtime.
//!
//! Elements are addressed with an `[usize; N]` index via [`Index`]/[`IndexMut`].
//! The two-dimensional case is specialized with extra conveniences — `(i, j)`
//! and whole-row indexing, row/column iterators, transpose, and rotations.
//!
//! # Why this instead of the alternatives?
//!
//! ## vs. nested vectors (`Vec<Vec<T>>`)
//!
//! The obvious alternative for a 2-D grid is `Vec<Vec<T>>` (and deeper nesting
//! for higher ranks). A single flat `Vec<T>` with computed offsets is preferred
//! because:
//!
//! - **One allocation, not `1 + rows + …`.** Nested vectors allocate an outer
//!   vector plus one inner vector per row, scattering rows across the heap.
//!   `MdArray` does a single allocation, so construction and drop are typically
//!   cheaper.
//! - **Cache locality.** All elements are contiguous, so row-major traversal is
//!   a linear memory scan — typically friendlier to the cache and prefetcher
//!   than chasing a pointer per row. Memory access is often the dominant cost
//!   in tight grid loops.
//! - **No per-access indirection.** Indexing is one offset computation plus one
//!   load, versus two dependent loads (`outer[i]` then `[j]`) for nested vecs.
//! - **Whole-array operations are trivial.** [`as_slice`](MdArray::as_slice),
//!   [`iter`](MdArray::iter), and [`fill`](MdArray::fill) act on the entire
//!   buffer at once; with nested vecs they would need a manual loop per row.
//! - **Shape is rectangular by construction.** Jagged rows are impossible, so
//!   the dimensions are a single source of truth and never drift out of sync.
//!
//! ## vs. fixed-size native arrays (`[[T; C]; R]`)
//!
//! A native nested array shares the contiguous, single-block layout, but its
//! lengths are part of the *type* and so must be compile-time constants:
//!
//! - **Runtime dimensions.** Contest inputs give you `n`, `m` (and possibly the
//!   rank's lengths) at runtime; `[[T; C]; R]` needs `R`/`C` known at compile
//!   time. `MdArray` takes the lengths as ordinary `usize` values.
//! - **Heap-allocated, won't blow the stack.** A `[[i64; 1000]; 1000]` is ~8 MB
//!   living on the stack — an easy stack overflow. `MdArray`'s buffer is on the
//!   heap, so large grids are fine.
//! - **One type for every shape.** A single `MdArray<T, N>` covers all lengths
//!   (and, via `N`, all ranks), instead of a distinct array type per shape.
//!
//! Native arrays still win when the dimensions genuinely are small compile-time
//! constants: they need no heap allocation and the sizes are checked statically.
//! `MdArray` targets the common runtime-shaped case.
//!
//! ```
//! use md_array::MdArray;
//!
//! let mut a = MdArray::new([2, 3], 0); // 2 rows, 3 cols, zero-filled
//! a[[1, 2]] = 7;
//! assert_eq!(a[[1, 2]], 7);
//! assert_eq!(a.dim_len(0), 2);
//! ```

// Compiles and runs the README's code blocks as doctests, without rendering it.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// Private numeric-cast helper used by [`MdArray::iota`].
///
/// Sealed: the trait is `pub` only so it can appear in a public bound, but it
/// lives in a private module and cannot be named or implemented downstream.
mod cast {
    /// Builds `Self` from a `T` with the semantics of `t as Self` (may truncate
    /// or wrap; never panics).
    pub trait Cast<T>: Copy {
        fn of(t: T) -> Self;
    }

    macro_rules! impl_cast_from_isize {
        ($($t: ty)+) => {
            $(impl Cast<isize> for $t {
                fn of(t: isize) -> Self {
                    t as $t
                }
            })+
        };
    }

    impl_cast_from_isize!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize);
}

use cast::Cast;
use std::ops::Range;
use std::vec::IntoIter;
use std::{
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

/// A dense `N`-dimensional array backed by a contiguous row-major [`Vec<T>`].
///
/// `dims_len[k]` is the length of dimension `k`; the backing `data` holds
/// exactly `dims_len.iter().product()` elements. See the [module docs] for the
/// layout convention and how this compares to nested `Vec`s and fixed-size
/// native arrays.
///
/// [module docs]: crate
#[derive(Clone, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
pub struct MdArray<T, const N: usize> {
    dims_len: [usize; N],
    data: Vec<T>,
}

impl<T, const N: usize> MdArray<T, N> {
    /// Maps a multi-dimensional index to its flat offset in `data`, encoding the
    /// row-major layout. Inverse of [`multi_index`](Self::multi_index).
    #[inline]
    fn flat_index(&self, dims_i: [usize; N]) -> usize {
        dims_i.iter().zip(&self.dims_len).fold(0, |i, (&di, &dn)| {
            // Deliberately `debug_assert!`: this is the hot path, and release
            // builds must not pay for a compare per dimension. See the `Index`
            // docs for what that means for out-of-range components.
            debug_assert!(
                di < dn,
                "index {di} out of bounds for dimension of length {dn}"
            );
            i * dn + di
        })
    }

    /// Decodes a flat offset back into its multi-dimensional index. Inverse of
    /// [`flat_index`](Self::flat_index).
    #[inline]
    fn multi_index(&self, mut at: usize) -> [usize; N] {
        let mut dims_i = [0usize; N];
        for k in (0..N).rev() {
            dims_i[k] = at % self.dims_len[k];
            at /= self.dims_len[k];
        }
        dims_i
    }

    /// Returns the length of dimension `i`.
    pub fn dim_len(&self, i: usize) -> usize {
        self.dims_len[i]
    }

    /// Iterates over all elements in row-major order.
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Mutably iterates over all elements in row-major order.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Borrows the whole backing buffer as a flat slice, in row-major order.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrows the whole backing buffer as a flat slice, in row-major
    /// order. The length is fixed, so the shape cannot be desynced.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Builds an array from an existing row-major buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `dims_len.iter().product()`.
    pub fn with_data(dims_len: [usize; N], data: Vec<T>) -> Self {
        assert!(dims_len.iter().product::<usize>() == data.len());
        Self { dims_len, data }
    }

    /// Fills the array with consecutive values `init, init + 1, …` in row-major
    /// order, each cast to `T`.
    ///
    /// Counting is done in `isize` and cast with `as`, so the usual `as`
    /// caveats apply (wrap/truncate for narrow `T`). Mainly useful for tests and
    /// quick fixtures.
    ///
    /// ```
    /// use md_array::MdArray;
    ///
    /// let a = MdArray::<i64, 2>::iota([2, 3], 0);
    /// assert_eq!(a.as_slice(), &[0, 1, 2, 3, 4, 5]);
    /// ```
    pub fn iota(dims_len: [usize; N], init: isize) -> Self
    where
        T: Cast<isize>,
    {
        let mut count = init;
        Self::with_gen(dims_len, |_| {
            let value = T::of(count);
            count += 1;
            value
        })
    }

    /// Builds an array by calling `f` once per cell, in row-major order, passing
    /// the cell's multi-dimensional index. If any dimension has length zero the
    /// array is empty and `f` is never called.
    ///
    /// ```
    /// use md_array::MdArray;
    ///
    /// let a = MdArray::with_gen([2, 3], |[i, j]| i * 10 + j);
    /// assert_eq!(a[[1, 2]], 12);
    /// ```
    pub fn with_gen<F>(dims_len: [usize; N], mut f: F) -> Self
    where
        F: FnMut([usize; N]) -> T,
    {
        let len = dims_len.iter().product();
        let mut data = Vec::with_capacity(len);
        let mut dims_i = [0usize; N];
        for _ in 0..len {
            data.push(f(dims_i));
            for k in (0..N).rev() {
                dims_i[k] += 1;
                if dims_i[k] < dims_len[k] {
                    break;
                }
                dims_i[k] = 0;
            }
        }
        Self { dims_len, data }
    }

    /// Returns the index of the first element (in row-major order) matching
    /// `predicate`, or `None` if there is none.
    pub fn position<P>(&self, predicate: P) -> Option<[usize; N]>
    where
        P: FnMut(&T) -> bool,
    {
        self.data
            .iter()
            .position(predicate)
            .map(|i| self.multi_index(i))
    }

    /// Returns the indices of all elements matching `predicate`, in row-major
    /// order.
    pub fn positions<P>(
        &self,
        mut predicate: P,
    ) -> impl DoubleEndedIterator<Item = [usize; N]> + use<'_, P, T, N>
    where
        P: FnMut(&T) -> bool,
    {
        self.data
            .iter()
            .enumerate()
            .filter_map(move |(i, t)| predicate(t).then_some(self.multi_index(i)))
    }

    /// Iterates over every multi-dimensional index of the array in row-major
    /// order, without borrowing the elements.
    ///
    /// Because the returned iterator captures only a copy of the dimensions (not
    /// `self`), the array may be mutated through it during iteration:
    ///
    /// ```
    /// use md_array::MdArray;
    ///
    /// let mut a = MdArray::new([2, 2], 0);
    /// for [i, j] in a.indices() {
    ///     a[[i, j]] = i * 2 + j; // `a` is free to borrow mutably here
    /// }
    /// assert_eq!(a.as_slice(), &[0, 1, 2, 3]);
    /// ```
    pub fn indices(
        &self,
    ) -> impl DoubleEndedIterator<Item = [usize; N]> + ExactSizeIterator + use<T, N> {
        let dims = self.dims_len;
        let size = dims.iter().product::<usize>();
        (0..size).map(move |at| {
            let mut idx = [0usize; N];
            let mut at = at;
            for k in (0..N).rev() {
                idx[k] = at % dims[k];
                at /= dims[k];
            }
            idx
        })
    }
}

impl<T: Clone, const N: usize> MdArray<T, N> {
    /// Builds an array of the given shape with every cell cloned from `value`.
    pub fn new(dims_len: [usize; N], value: T) -> Self {
        Self {
            dims_len,
            data: vec![value; dims_len.iter().product()],
        }
    }

    /// Overwrites every element with a clone of `value`, keeping the shape.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

/// Returns an empty array (all dimensions zero, no elements).
impl<T, const N: usize> Default for MdArray<T, N> {
    fn default() -> Self {
        Self {
            dims_len: [0usize; N],
            data: Default::default(),
        }
    }
}

/// Indexes by a full multi-dimensional index `[i0, …, i_{N-1}]`.
///
/// # Panics
///
/// In debug builds, panics if any component is out of bounds for its dimension.
///
/// In release builds that per-component check is omitted for speed, so an
/// out-of-range component is **not** reliably caught: it may land on a different
/// valid cell (for a `2 × 3` array, `[0, 3]` reads the cell at `[1, 0]`) and
/// silently return the wrong element. Only an offset past the end of the whole
/// buffer panics. This is always memory-safe, never undefined behaviour.
impl<T, const N: usize> Index<[usize; N]> for MdArray<T, N> {
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.data[self.flat_index(index)]
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for MdArray<T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let at = self.flat_index(index);
        &mut self.data[at]
    }
}

/// Borrows the row-major backing buffer as a slice.
impl<T, const N: usize> AsRef<[T]> for MdArray<T, N> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

/// Mutably borrows the row-major backing buffer as a slice. The length is fixed,
/// so the stored dimensions cannot be desynced.
impl<T, const N: usize> AsMut<[T]> for MdArray<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Consumes the array, yielding owned elements in row-major order.
impl<T, const N: usize> IntoIterator for MdArray<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Borrows the array, yielding element references in row-major order.
impl<'a, T, const N: usize> IntoIterator for &'a MdArray<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Two-dimensional specializations: row/column access, in-place swaps, and
/// shape-changing rotations and transpose.
impl<T> MdArray<T, 2> {
    /// Number of rows (length of dimension 0).
    #[inline]
    pub fn rows_len(&self) -> usize {
        self.dims_len[0]
    }

    /// Number of columns (length of dimension 1).
    #[inline]
    pub fn cols_len(&self) -> usize {
        self.dims_len[1]
    }

    /// Range `0..rows_len()`, convenient for `for r in a.rows()`.
    #[inline]
    pub fn rows(&self) -> Range<usize> {
        0..self.rows_len()
    }

    /// Range `0..cols_len()`, convenient for `for c in a.cols()`.
    #[inline]
    pub fn cols(&self) -> Range<usize> {
        0..self.cols_len()
    }

    /// Whether the grid is square (`rows_len() == cols_len()`).
    #[inline]
    pub fn is_square(&self) -> bool {
        self.dims_len[0] == self.dims_len[1]
    }

    /// Iterates over the elements of row `at`, left to right.
    ///
    /// # Panics
    ///
    /// Panics if `at >= rows_len()`.
    pub fn row(&self, at: usize) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        assert!(at < self.dims_len[0]);
        self[at].iter()
    }

    /// Mutably iterates over the elements of row `at`, left to right.
    ///
    /// # Panics
    ///
    /// Panics if `at >= rows_len()`.
    pub fn row_mut(
        &mut self,
        at: usize,
    ) -> impl DoubleEndedIterator<Item = &mut T> + ExactSizeIterator {
        assert!(at < self.dims_len[0]);
        self[at].iter_mut()
    }

    /// Iterates over the elements of column `at`, top to bottom.
    ///
    /// # Panics
    ///
    /// Panics if `at >= cols_len()`.
    pub fn col(&self, at: usize) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        assert!(at < self.dims_len[1]);
        self.data.iter().skip(at).step_by(self.dims_len[1])
    }

    /// Mutably iterates over the elements of column `at`, top to bottom.
    ///
    /// # Panics
    ///
    /// Panics if `at >= cols_len()`.
    pub fn col_mut(
        &mut self,
        at: usize,
    ) -> impl DoubleEndedIterator<Item = &mut T> + ExactSizeIterator {
        assert!(at < self.dims_len[1]);
        self.data.iter_mut().skip(at).step_by(self.dims_len[1])
    }

    /// Swaps the two cells at `(row, col)` coordinates `(r1, c1)` and `(r2, c2)`.
    ///
    /// # Panics
    ///
    /// Panics if either coordinate is out of bounds.
    pub fn swap(&mut self, (r1, c1): (usize, usize), (r2, c2): (usize, usize)) {
        assert!(r1 < self.dims_len[0]);
        assert!(r2 < self.dims_len[0]);
        assert!(c1 < self.dims_len[1]);
        assert!(c2 < self.dims_len[1]);
        self.data
            .swap(r1 * self.dims_len[1] + c1, r2 * self.dims_len[1] + c2);
    }

    /// Swaps rows `r1` and `r2` in place. A no-op if `r1 == r2`.
    ///
    /// # Panics
    ///
    /// Panics if either row index is out of bounds.
    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        assert!(r1 < self.dims_len[0]);
        assert!(r2 < self.dims_len[0]);
        if r1 == r2 {
            return;
        }
        let cols = self.cols_len();
        let (lo, hi) = (r1.min(r2), r1.max(r2));
        let (head, tail) = self.data.split_at_mut(hi * cols);
        head[lo * cols..(lo + 1) * cols].swap_with_slice(&mut tail[..cols]);
    }

    /// Appends one row to the bottom of the grid, growing `rows_len()` by one.
    ///
    /// `iter` must yield exactly `cols_len()` elements so the grid stays
    /// rectangular. The one exception is a `0 × 0` grid (such as
    /// [`Default::default()`]): its first row sets the column count.
    ///
    /// # Panics
    ///
    /// Panics if `iter`'s length differs from `cols_len()`; the grid is left
    /// unchanged.
    ///
    /// ```
    /// use md_array::MdArray;
    ///
    /// let mut a = MdArray::<usize, 2>::iota([1, 3], 1); // [[1, 2, 3]]
    /// a.push_row([4, 5, 6]);
    /// assert_eq!(a.rows_len(), 2);
    /// assert_eq!(a[1], [4, 5, 6]);
    ///
    /// // Building a grid row by row from an empty one:
    /// let mut b = MdArray::<usize, 2>::default();
    /// b.push_row([1, 2]);
    /// b.push_row([3, 4]);
    /// assert_eq!((b.rows_len(), b.cols_len()), (2, 2));
    /// ```
    pub fn push_row<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let before = self.data.len();
        self.data.extend(iter);
        let len = self.data.len() - before;
        if self.dims_len == [0, 0] {
            self.dims_len[1] = len;
        } else if len != self.dims_len[1] {
            self.data.truncate(before);
            panic!("iter size is not cols");
        }
        self.dims_len[0] += 1;
    }

    /// Mirrors the grid across its horizontal mid-line (flips top↔bottom),
    /// mapping `(i, j) -> (rows - i - 1, j)`.
    ///
    /// Done in place by swapping mirrored rows; the shape is unchanged. Consumes
    /// and returns `self` (no `Clone` required).
    pub fn reflect_horizontally(mut self) -> Self {
        let rows = self.rows_len();
        let m = rows >> 1;
        for r in 0..m {
            self.swap_rows(r, rows - r - 1);
        }
        self
    }

    /// Mirrors the grid across its vertical mid-line (flips left↔right),
    /// mapping `(i, j) -> (i, cols - j - 1)`.
    ///
    /// Done in place by reversing each row; the shape is unchanged. Consumes and
    /// returns `self` (no `Clone` required).
    pub fn reflect_vertically(mut self) -> Self {
        // `chunks_exact_mut(0)` panics; with zero columns there is no data anyway.
        let cols = self.cols_len().max(1);
        self.data
            .chunks_exact_mut(cols)
            .for_each(|row| row.reverse());
        self
    }

    /// Returns the grid rotated 90° clockwise, mapping `(i, j) -> (j, rows - i - 1)`.
    ///
    /// An `R × C` grid becomes `C × R`. Consumes `self` and moves each element
    /// into the result (no `Clone` required).
    pub fn rotate_clockwise(self) -> Self {
        let rows = self.dims_len[0];
        self.remap(|i, j| j * rows + rows - i - 1)
    }

    /// Returns the grid rotated 90° counter-clockwise, mapping
    /// `(i, j) -> (cols - j - 1, i)`.
    ///
    /// An `R × C` grid becomes `C × R`. Consumes `self` and moves each element
    /// into the result (no `Clone` required).
    pub fn rotate_counter_clockwise(self) -> Self {
        let rows = self.dims_len[0];
        let cols = self.dims_len[1];
        self.remap(|i, j| (cols - j - 1) * rows + i)
    }

    /// Returns the transpose, mapping `(i, j) -> (j, i)`; an `R × C` grid
    /// becomes `C × R`.
    ///
    /// Consumes `self` and moves each element into place (no `Clone` required).
    /// A square grid is transposed in place by swapping mirrored cells;
    /// otherwise the elements are scattered into a fresh buffer.
    pub fn transpose(self) -> Self {
        if self.is_square() {
            let n = self.dims_len[0];
            let mut a = self;
            for i in 0..n {
                for j in i + 1..n {
                    a.data.swap(i * n + j, j * n + i);
                }
            }
            a
        } else {
            let rows = self.dims_len[0];
            self.remap(|i, j| j * rows + i)
        }
    }

    /// Scatters every element into a fresh `cols × rows` buffer, moving the cell
    /// at `(i, j)` to the flat offset returned by `to(i, j)`. Shared engine
    /// behind [`transpose`](Self::transpose),
    /// [`rotate_clockwise`](Self::rotate_clockwise), and
    /// [`rotate_counter_clockwise`](Self::rotate_counter_clockwise) — each just
    /// supplies its own coordinate map.
    ///
    /// Consumes `self` and moves each element exactly once (no `Clone`). `to`
    /// must be a bijection onto `0..rows * cols`: every destination offset has to
    /// be produced exactly once, or the result buffer is left with uninitialized
    /// or doubly-written cells.
    ///
    /// Writes into the vector's spare capacity (`&mut [MaybeUninit<T>]`) rather
    /// than building the destination in order, since the mapping visits
    /// destination offsets out of sequence. `set_len` is called only after every
    /// cell has been written, so the `Vec` never claims uninitialized elements. A
    /// panic part-way (e.g. from `to`) merely leaks the moved elements.
    fn remap(self, to: impl Fn(usize, usize) -> usize) -> Self {
        let rows = self.dims_len[0];
        let cols = self.dims_len[1];
        let size = rows * cols;
        let mut data = Vec::with_capacity(size);
        let spare = data.spare_capacity_mut();
        for (id, e) in self.data.into_iter().enumerate() {
            // Bounds-checked: an out-of-range `to` panics instead of writing OOB.
            spare[to(id / cols, id % cols)].write(e);
        }
        // SAFETY: `to` is a bijection onto `0..size` (see above), so all `size`
        // cells of the spare capacity were initialized exactly once.
        unsafe { data.set_len(size) };
        Self {
            dims_len: [cols, rows],
            data,
        }
    }
}

/// Indexes a whole row by its number, yielding it as a contiguous slice.
///
/// # Panics
///
/// In debug builds, panics if `index >= rows_len()`. Release builds omit that
/// check (as for the `[usize; N]` index): an out-of-range row still panics from the underlying slice bounds, except in
/// a grid with zero columns, where it yields an empty slice.
impl<T> Index<usize> for MdArray<T, 2> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(
            index < self.rows_len(),
            "row index {index} out of bounds for {} rows",
            self.rows_len()
        );
        &self.data[self.dims_len[1] * index..self.dims_len[1] * (index + 1)]
    }
}

impl<T> IndexMut<usize> for MdArray<T, 2> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.dims_len[1] * index..self.dims_len[1] * (index + 1)]
    }
}

/// Indexes a single cell by `(row, col)`; an ergonomic alias for `[[i, j]]`.
///
/// Bounds checking is identical to `[[i, j]]`: debug builds panic if
/// `i >= rows_len()` or `j >= cols_len()`, release builds do not check each
/// component (see the `[usize; N]` index).
impl<T> Index<(usize, usize)> for MdArray<T, 2> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self[[i, j]]
    }
}

impl<T> IndexMut<(usize, usize)> for MdArray<T, 2> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self[[i, j]]
    }
}

#[cfg(test)]
mod tests {
    use super::MdArray;

    #[test]
    fn rows_and_cols_in_2d_array() {
        let a = MdArray::<usize, 2>::iota([3, 4], 1);
        assert!(a.row(1).copied().eq(5..=8));
        assert!(a.col(2).copied().eq([3, 7, 11]));
    }

    #[test]
    fn mut_rows_and_cols_in_2d_array() {
        let mut a = MdArray::<usize, 2>::iota([3, 4], 1);
        a.row_mut(1).for_each(|x| *x += 1);
        assert!(a.row(1).copied().eq(6..=9));
        a.col_mut(1).for_each(|x| *x *= 2);
        assert!(a.col(1).copied().eq([4, 14, 20]));
    }

    #[test]
    fn swap_2d() {
        let mut a = MdArray::<usize, 2>::iota([3, 4], 1);
        a.swap((0, 2), (2, 1));
        assert_eq!(a[(0, 2)], 10);
        assert_eq!(a[(2, 1)], 3);
        assert_eq!(a[0], [1, 2, 10, 4]);
        assert_eq!(a[2], [9, 3, 11, 12]);
    }

    #[test]
    fn swap_rows() {
        let mut a = MdArray::<usize, 2>::iota([4, 4], 1);
        a.swap_rows(1, 3);
        assert_eq!(a[1], [13, 14, 15, 16]);
        assert_eq!(a[3], [5, 6, 7, 8]);
        a.swap_rows(0, 2);
        assert_eq!(a[0], [9, 10, 11, 12]);
        assert_eq!(a[2], [1, 2, 3, 4]);
    }

    #[test]
    fn transpose() {
        let a = MdArray::<usize, 2>::iota([3, 2], 1);
        let a = a.transpose();
        assert_eq!(a[0], [1, 3, 5]);
        assert_eq!(a[1], [2, 4, 6]);
    }

    #[test]
    fn transpose_square() {
        let a = MdArray::<usize, 2>::iota([3, 3], 1);
        let a = a.transpose();
        assert_eq!(a[0], [1, 4, 7]);
        assert_eq!(a[1], [2, 5, 8]);
        assert_eq!(a[2], [3, 6, 9]);
    }

    #[test]
    fn rotate_clockwise() {
        let a = MdArray::<usize, 2>::iota([2, 3], 1);
        let a = a.rotate_clockwise();
        assert_eq!(a[0], [4, 1]);
        assert_eq!(a[1], [5, 2]);
        assert_eq!(a[2], [6, 3]);
    }

    #[test]
    fn rotate_counter_clockwise() {
        let a = MdArray::<usize, 2>::iota([2, 3], 1);
        let a = a.rotate_counter_clockwise();
        assert_eq!(a[0], [3, 6]);
        assert_eq!(a[1], [2, 5]);
        assert_eq!(a[2], [1, 4]);
    }

    #[test]
    fn rotations_and_transpose_move_non_copy_elements() {
        let a = MdArray::with_gen([2, 3], |[i, j]| format!("{i}{j}"));
        let t = a.clone().transpose();
        assert_eq!(t[0], ["00", "10"]);
        assert_eq!(t[2], ["02", "12"]);
        assert_eq!(a.clone().rotate_clockwise()[0], ["10", "00"]);
        assert_eq!(a.rotate_counter_clockwise()[0], ["02", "12"]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "out of bounds")]
    fn index_component_out_of_bounds_panics_in_debug() {
        // `[0, 3]` aliases `[1, 0]` in the flat buffer; only the debug check sees it.
        let a = MdArray::<usize, 2>::iota([2, 3], 0);
        let _ = a[[0, 3]];
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "out of bounds")]
    fn index_tuple_out_of_bounds_panics_in_debug() {
        let a = MdArray::<usize, 2>::iota([2, 3], 0);
        let _ = a[(0, 3)];
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "out of bounds")]
    fn index_row_out_of_bounds_panics_in_debug() {
        let a = MdArray::<usize, 2>::iota([2, 3], 0);
        let _ = &a[2];
    }

    #[test]
    #[should_panic]
    fn row_out_of_bounds_always_panics() {
        let a = MdArray::<usize, 2>::iota([2, 3], 0);
        let _ = a.row(2).count();
    }

    #[test]
    fn empty_arrays() {
        for dims in [[0, 3], [3, 0], [0, 0]] {
            let a = MdArray::<i32, 2>::with_gen(dims, |_| 1);
            assert!(a.as_slice().is_empty());
            assert_eq!(a.position(|_| true), None);
            assert_eq!(a.indices().count(), 0);
            assert_eq!(a.clone().reflect_vertically().reflect_horizontally(), a);
            let t = a.clone().transpose();
            assert_eq!((t.rows_len(), t.cols_len()), (dims[1], dims[0]));
            let r = a.rotate_clockwise();
            assert_eq!((r.rows_len(), r.cols_len()), (dims[1], dims[0]));
        }
        assert_eq!(MdArray::new([2, 0], 0).row(1).count(), 0);
    }

    #[test]
    fn as_ref_and_as_mut_slice() {
        let mut a = MdArray::<usize, 2>::iota([2, 2], 1);
        a.as_mut()[0] = 9;
        a.as_mut_slice()[1] = 8;
        let s: &[usize] = a.as_ref();
        assert_eq!(s, [9, 8, 3, 4]);
    }

    #[test]
    fn push_row_onto_default_sets_cols() {
        let mut a = MdArray::<usize, 2>::default();
        a.push_row([1, 2, 3]);
        a.push_row([4, 5, 6]);
        assert_eq!((a.rows_len(), a.cols_len()), (2, 3));
        assert_eq!(a[1], [4, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "iter size is not cols")]
    fn push_row_wrong_length_panics() {
        let mut a = MdArray::<usize, 2>::default();
        a.push_row([1, 2, 3]);
        a.push_row([4, 5]);
    }

    #[test]
    fn push_row_wrong_length_leaves_grid_unchanged() {
        let mut a = MdArray::<usize, 2>::iota([1, 3], 1);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.push_row([4, 5])));
        assert!(r.is_err());
        assert_eq!(a, MdArray::<usize, 2>::iota([1, 3], 1));
    }

    #[test]
    fn reflect_vertically() {
        let a = MdArray::<usize, 2>::iota([2, 3], 1);
        let a = a.reflect_vertically();
        assert_eq!(a[0], [3, 2, 1]);
        assert_eq!(a[1], [6, 5, 4]);
    }

    #[test]
    fn reflect_horizontally() {
        let a = MdArray::<usize, 2>::iota([3, 2], 1);
        let a = a.reflect_horizontally();
        assert_eq!(a[0], [5, 6]);
        assert_eq!(a[1], [3, 4]);
        assert_eq!(a[2], [1, 2]);
    }

    #[test]
    fn position() {
        let a = MdArray::<usize, 2>::iota([3, 4], 1);
        let s = a.position(|&x| x == 7);
        assert_eq!(s, Some([1, 2]));
        let s = a.position(|&x| x == 20);
        assert!(s.is_none());
    }

    #[test]
    fn positions() {
        let a = MdArray::<usize, 3>::iota([2, 2, 2], 1);
        let e = a.positions(|&x| x.is_multiple_of(2));
        let o = a.positions(|&x| !x.is_multiple_of(2));
        assert!(e.eq([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]));
        assert!(o.eq([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]));
    }

    #[test]
    fn indices_2d() {
        let a = MdArray::new([2, 2], 0);
        assert!(a.indices().eq([[0, 0], [0, 1], [1, 0], [1, 1]]));
    }

    #[test]
    fn indices_3d() {
        let a = MdArray::new([2, 2, 2], 0);
        let mut indices = a.indices();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_eq!(indices.next(), Some([i, j, k]));
                }
            }
        }
    }

    #[test]
    fn indices_4d() {
        let a = MdArray::new([2, 2, 2, 2], 0);
        let mut indices = a.indices();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for z in 0..2 {
                        assert_eq!(indices.next(), Some([i, j, k, z]));
                    }
                }
            }
        }
    }
}
