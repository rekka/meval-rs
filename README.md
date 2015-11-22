# meval-rs

[![Build Status](https://travis-ci.org/rekka/meval-rs.svg?branch=master)](https://travis-ci.org/rekka/meval-rs)

This [Rust] crate provides a simple math expression parsing and evaluation. Its main goal is to
be convenient to use, while allowing for some flexibility.

## Documentation

[Full API documentation](http://rekka.github.io/meval-rs/meval/index.html)

## Installation

Simply add the corresponding entry to your `Cargo.toml` dependency list:

```toml
[dependencies]
meval = { git = "https://github.com/rekka/meval-rs" }
```

and add this to your crate root:

```rust
extern crate meval;
```

## Simple examples

```rust
extern crate meval;

fn main() {
    let r = meval::eval_str("1 + 2").unwrap();

    println!("1 + 2 = {}", r);
}
```

Need to define a Rust function from an expression? No problem, use
[`Expr`](http://rekka.github.io/meval-rs/meval/struct.Expr.html) for
this and more:

```rust
extern crate meval;

fn main() {
    let expr = meval::Expr::from_str("sin(pi * x)").unwrap();
    let func = expr.bind("x").unwrap();

    let vs: Vec<_> = (0..100+1).map(|i| func(i as f64 / 100.)).collect();

    println!("sin(pi * x), 0 <= x <= 1: {:?}", vs);
}
```

[`Expr::bind`](http://rekka.github.io/meval-rs/meval/struct.Expr.html#method.bind)
returns a boxed closure that is slightly less convenient than an unboxed
closure since `Box<Fn(f64) -> f64>` does not implement `FnOnce`, `Fn` or
`FnMut`. So to use it directly as a function argument where a closure is
expected, it has to be manually dereferenced:

```rust
let func = meval::Expr::from_str("x").unwrap().bind("x").unwrap();
let r = Some(2.).map(&*func);
```

## Supported expressions

`meval` supports basic mathematical operations on floating point numbers:

- binary operators: `+`, `-`, `*`, `/`, `^` (power)
- unary operators: `+`, `-`

It supports custom variables like `x`, `weight`, `C_0`, etc. A variable must start with
`[a-zA-Z_]` and can contain only `[a-zA-Z0-9_]`.

Build-in functions currently supported (implemented using functions of the same name in [Rust
std library][std-float]):

- `sqrt`, `abs`
- `exp`, `ln`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- `floor`, `ceil`, `round`
- `signum`

Build-in constants:

- `pi`
- `e`

## Related projects

This is a toy project of mine for learning Rust, and to be hopefully useful when writing
command line scripts. For other similar projects see:

- [rodolf0/tox](https://github.com/rodolf0/tox)

[Rust]: https://www.rust-lang.org/
[std-float]: http://doc.rust-lang.org/stable/std/primitive.f64.html

## License

This project is dual-licensed under the Unlicense and MIT licenses.

You may use this code under the terms of either license.
