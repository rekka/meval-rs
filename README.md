# meval-rs

This [Rust] crate provides a simple math expression parser. Its main goal is to be convenient
to use, while allowing for some flexibility.

## Simple examples

```rust
extern crate meval;

fn main() {
    let r = meval::eval_str("1 + 2").unwrap();

    println!("1 + 2 = {}", r);
}
```

Need to define a Rust function from an expression? No problem, use [`Expr`](struct.Expr.html)
for this and more:

```rust
extern crate meval;

fn main() {
    let expr = meval::Expr::from_str("sin(pi * x)").unwrap();
    let func = expr.bind("x").unwrap();

    let vs: Vec<_> = (0..100+1).map(|i| func(i as f64 / 100.)).collect();

    println!("sin(pi * x), 0 <= x <= 1: {:?}", vs);
}
```

## Supported expressions

`meval` supports basic mathematical operations on floating point numbers:

- binary operators: `+`, `-`, `*`, `/`, `^` (power)
- unary operators: `+`, `-`

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

