# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MdArray<T, N>`: a dense, fixed-rank multi-dimensional array backed by a
  single contiguous row-major `Vec<T>`.
- 2-D conveniences: `(i, j)` and whole-row indexing, row/column iterators,
  `swap`, `swap_rows`, `push_row`, `transpose`, 90° rotations and reflections.
