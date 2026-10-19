# md-array

[![CI](https://github.com/omar-s-ta/md-array/actions/workflows/ci.yml/badge.svg)](https://github.com/omar-s-ta/md-array/actions/workflows/ci.yml)

A small, dependency-free, dense **multi-dimensional array** for Rust, aimed at
competitive programming and other grid-heavy code.

`MdArray<T, N>` stores `N`-dimensional data in one contiguous `Vec<T>` in
**row-major** order. The rank `N` is a const generic; the length of each
dimension is chosen at runtime.

```sh
cargo add md-array
```

```rust
use md_array::MdArray;

let mut a = MdArray::new([2, 3], 0); // 2 rows, 3 columns, zero-filled
a[[1, 2]] = 7;
assert_eq!(a[[1, 2]], 7);
assert_eq!(a[(1, 2)], 7); // 2-D convenience
assert_eq!(a[1], [0, 0, 7]); // a whole row, as a slice

let t = a.transpose(); // 3 rows, 2 columns
assert_eq!(t[2], [0, 7]);
```

## Building an array

```rust
use md_array::MdArray;

// Every cell cloned from one value.
let a = MdArray::new([2, 3], 0);

// One call per cell, in row-major order, given the cell's index.
let b = MdArray::with_gen([2, 3], |[i, j]| i * 10 + j);
assert_eq!(b[[1, 2]], 12);

// Counting up from a start value: handy for tests and fixtures.
let c = MdArray::<i64, 2>::iota([2, 3], 1);
assert_eq!(c.as_slice(), &[1, 2, 3, 4, 5, 6]);

// From an existing row-major buffer (panics if the length doesn't match).
let d = MdArray::with_data([2, 3], vec![1, 2, 3, 4, 5, 6]);
assert_eq!(d[1], [4, 5, 6]);
```

## Bounds checking

Indexing with `[i, j, …]` or `(i, j)` is bounds-checked **in debug builds
only**, to keep release builds fast. In release builds an out-of-range component
is not reliably caught: on a `2 × 3` array, `[0, 3]` reads the cell at `[1, 0]`
instead of panicking. This is memory-safe, but it can silently return the wrong
element. Methods such as `row`, `col` and `swap` always check their arguments.
See the `Index` docs for the details.

## Why

- **One allocation** instead of `Vec<Vec<T>>`'s one per row, so it is typically
  faster to build and friendlier to the cache.
- **Runtime dimensions** and **heap storage**, unlike `[[T; C]; R]`, so large
  grids don't overflow the stack.
- **Rectangular by construction**: there are no jagged rows.
- **2-D helpers** you otherwise rewrite for every grid problem: row and column
  iterators, `transpose`, `rotate_clockwise`, `rotate_counter_clockwise`,
  `reflect_horizontally`, `reflect_vertically`, `swap_rows`, `push_row`.
- **Search helpers**: `position` and `positions` return `[usize; N]` indices;
  `indices()` iterates every index without borrowing the array.

## Not for everything

`md-array` is deliberately minimal. It has no views, slicing, broadcasting or
numerics. If you need those, use [`ndarray`](https://crates.io/crates/ndarray);
for a plain 2-D grid, [`grid`](https://crates.io/crates/grid) is also a good
choice.

## Dependencies and Rust version

- No required dependencies. If optional features ever pull in crates, they will
  be off by default.
- The crate uses edition 2024, so it needs Rust 1.85 or newer.
- The version is `0.x`; the API may change between minor releases.

## Source

The source and issue tracker are at
<https://github.com/omar-s-ta/md-array>.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
