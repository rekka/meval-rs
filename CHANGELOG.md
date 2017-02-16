# 0.0.8

- added serde deserialization
- api change: `Expr::eval` is now `Expr::eval_with_context`
- `Expr` implements `FromStr`: use it as "1 + 2".parse::<Expr>()
- added `de::as_f64` for convenient deserialization
- `Context` is now `Clone`
