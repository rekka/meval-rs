//! Deserialization utilities.
use super::Expr;
use serde::de;
use serde::de::Error;
use serde::Deserialize;

/// Deserialize into [`Expr`](../struct.Expr.html) and then evaluate using `Expr::eval`.
///
/// # Example
///
/// ```rust
/// #[macro_use]
/// extern crate serde_derive;
/// extern crate toml;
/// extern crate meval;
/// use meval::{Expr, Context};
///
/// #[derive(Deserialize)]
/// struct Foo {
///     #[serde(deserialize_with = "meval::de::as_f64")]
///     x: f64,
/// }
///
/// fn main() {
///     let foo: Foo = toml::from_str(r#" x = "cos(1.)" "#).unwrap();
///     assert_eq!(foo.x, 1f64.cos());
///
///     let foo: Result<Foo, _> = toml::from_str(r#" x = "cos(x)" "#);
///     assert!(foo.is_err());
/// }
/// ```
///
/// See [crate root](../index.html#deserialization) for another example.
pub fn as_f64<'de, D: de::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Expr::deserialize(deserializer)?
        .eval()
        .map_err(D::Error::custom)
}
