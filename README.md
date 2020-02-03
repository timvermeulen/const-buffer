# const-buffer

[![crates.io](https://img.shields.io/crates/v/const-buffer)](https://crates.io/crates/const-buffer)
[![docs.rs](https://docs.rs/const-buffer/badge.svg)](https://docs.rs/const-buffer)
![rustc version](https://img.shields.io/badge/rustc-nightly-inactive)

A fixed-capacity memory buffer allocated on the stack using const generics.

This is a low-level utility, useful for implementing higher-level data structures such as fixed-capacity vectors and ring buffers. Since `ConstBuffer`'s main purpose is to build safe abstractions on top of, almost its entire API surface is `unsafe`.

`ConstBuffer` does not keep track of which elements are in an initialized state. Furthermore, in order to ensure optimal performance, **no bounds checks are performed** unless debug assertions are enabled. Any misuse of this crate leads to undefined behavior.

## Example usage

```rust
use const_buffer::ConstBuffer;

fn main() {
    let mut buffer = ConstBuffer::<u32, 10>::new();

    unsafe {
        buffer.copy_from_slice(2, &[10, 20, 30]);
        // [_, _, 10, 20, 30, _, _, _, _, _]
        buffer.write(5, 40);
        // [_, _, 10, 20, 30, 40, _, _, _, _]
        assert_eq!(buffer.get(3), &20);
        assert_eq!(buffer.get(2..6), &[10, 20, 30, 40]);

        buffer.copy_within(2..6, 0);
        // [10, 20, 30, 40, 30, 40, _, _, _, _]
        buffer.get_mut(1..4).reverse();
        // [10, 40, 30, 20, 30, 40, _, _, _, _]
        assert_eq!(buffer.get(..6), &[10, 40, 30, 20, 30, 40]);

        buffer.swap(3, 8);
        // [10, 40, 30, _, 30, 40, _, _, 20, _]
        assert_eq!(buffer.read(8), 20);
    }
}
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option. Any source contributions will be dual-licensed in the same way.
