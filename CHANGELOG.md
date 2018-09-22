# Unreleased

- `bind` and `bind_with_context` functions return unboxed closures
  (requires Rust 1.26).

# 0.1.1

- Implement `Default` for `Context`

# 0.1.0

- Support `serde-1.0.0`

# 0.0.9

- Bug #13: failed build with `serde` feature disabled

# 0.0.8

- added serde deserialization
- api change: `Expr::eval` is now `Expr::eval_with_context`
- `Expr` implements `FromStr`: use it as "1 + 2".parse::<Expr>()
- added `de::as_f64` for convenient deserialization
- `Context` is now `Clone`
