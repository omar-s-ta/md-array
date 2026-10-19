# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-19

### Added

- `MdArray<T, N>`: a dense, fixed-rank multi-dimensional array backed by a
  single contiguous row-major `Vec<T>`.
- Construction: `new`, `with_gen`, `iota` and `with_data`. Arrays with a
  zero-length dimension are allowed and empty.
- Indexing by `[usize; N]`, plus iteration (`iter`, `iter_mut`, `indices`),
  slice access (`as_slice`, `as_mut_slice`, `AsRef<[T]>`, `AsMut<[T]>`) and
  search (`position`, `positions`).
- 2-D conveniences: `(i, j)` and whole-row indexing, row/column iterators,
  `swap`, `swap_rows`, `push_row` (a `0 × 0` grid takes its column count from
  the first row), `transpose`, 90° rotations and reflections.

### Notes

- Indexing with `[i, j, …]` or `(i, j)` is bounds-checked in debug builds only;
  in release builds an out-of-range component may read a different cell instead
  of panicking. This is memory-safe. See the `Index` docs.
- Requires Rust 1.85 or newer (edition 2024).

[Unreleased]: https://github.com/omar-s-ta/md-array/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/omar-s-ta/md-array/releases/tag/v0.1.0
