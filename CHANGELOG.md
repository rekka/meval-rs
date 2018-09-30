# 0.2

- `bind` and `bind_with_context` functions return unboxed closures
  (requires Rust 1.26).

- [#17](https://github.com/rekka/meval-rs/pull/17)

    - Implement `bind4`, `bind4_with_context`, `bind5`,
      `bind5_with_context` and `bindn`.

    - Bugfix: Serialization from JSON now works as expected.

- Drop support for serialization from non-self-describing formats.

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
