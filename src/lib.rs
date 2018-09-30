//! This [Rust] crate provides a simple math expression parsing and evaluation. Its main goal is to
//! be convenient to use, while allowing for some flexibility. Currently works only with `f64`
//! types. A typical use case is the configuration of numerical computations in
//! Rust, think initial data and boundary conditions, via config files or command line arguments.
//!
//! # Documentation
//!
//! - [Full API documentation](https://docs.rs/meval)
//!
//! # Installation
//!
//! Simply add the corresponding entry to your `Cargo.toml` dependency list:
//!
//! ```toml
//! [dependencies]
//! meval = "0.2"
//! ```
//!
//! and add this to your crate root:
//!
//! ```rust
//! extern crate meval;
//! ```
//!
//!  **Requires Rust 1.26.**
//!
//! # Simple examples
//!
//! ```rust
//! extern crate meval;
//!
//! fn main() {
//!     let r = meval::eval_str("1 + 2").unwrap();
//!
//!     println!("1 + 2 = {}", r);
//! }
//! ```
//!
//! Need to define a Rust function from an expression? No problem, use [`Expr`][Expr]
//! for this and more:
//!
//! ```rust
//! extern crate meval;
//!
//! fn main() {
//!     let expr: meval::Expr = "sin(pi * x)".parse().unwrap();
//!     let func = expr.bind("x").unwrap();
//!
//!     let vs: Vec<_> = (0..100+1).map(|i| func(i as f64 / 100.)).collect();
//!
//!     println!("sin(pi * x), 0 <= x <= 1: {:?}", vs);
//! }
//! ```
//!
//! Custom constants and functions? Define a [`Context`][Context]!
//!
//! ```rust
//! use meval::{Expr, Context};
//!
//! let y = 1.;
//! let expr: Expr = "phi(-2 * zeta + x)".parse().unwrap();
//!
//! // create a context with function definitions and variables
//! let mut ctx = Context::new(); // built-ins
//! ctx.func("phi", |x| x + y)
//!    .var("zeta", -1.);
//! // bind function with a custom context
//! let func = expr.bind_with_context(ctx, "x").unwrap();
//! assert_eq!(func(2.), -2. * -1. + 2. + 1.);
//! ```
//!
//! For functions of 2, 3, and N variables use `Context::func2`, `Context::func3` and
//! `Context::funcn`,
//! respectively. See [`Context`][Context] for more options.
//!
//! If you need a custom function depending on mutable parameters, you will need to use a
//! [`Cell`](https://doc.rust-lang.org/stable/std/cell/struct.Cell.html):
//!
//! ```rust
//! use std::cell::Cell;
//! use meval::{Expr, Context};
//! let y = Cell::new(0.);
//! let expr: Expr = "phi(x)".parse().unwrap();
//!
//! let mut ctx = Context::empty(); // no built-ins
//! ctx.func("phi", |x| x + y.get());
//!
//! let func = expr.bind_with_context(ctx, "x").unwrap();
//! assert_eq!(func(2.), 2.);
//! y.set(3.);
//! assert_eq!(func(2.), 5.);
//! ```
//!
//! # Supported expressions
//!
//! `meval` supports basic mathematical operations on floating point numbers:
//!
//! - binary operators: `+`, `-`, `*`, `/`, `%` (remainder), `^` (power)
//! - unary operators: `+`, `-`
//!
//! It supports custom variables and functions like `x`, `weight`, `C_0`, `f(1)`, etc. A variable
//! or function name must start with `[a-zA-Z_]` and can contain only `[a-zA-Z0-9_]`. Custom
//! functions with a variable number of arguments are also supported.
//!
//! Build-ins (given by the context `Context::new()` and when no context provided) currently
//! supported:
//!
//! - functions implemented using functions of the same name in [Rust std library][std-float]:
//!
//!     - `sqrt`, `abs`
//!     - `exp`, `ln`
//!     - `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
//!     - `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
//!     - `floor`, `ceil`, `round`
//!     - `signum`
//!
//! - other functions:
//!
//!     - `max(x, ...)`, `min(x, ...)`: maximum and minimumum of 1 or more numbers
//!
//! - constants:
//!
//!     - `pi`
//!     - `e`
//!
//! # Deserialization
//!
//! [`Expr`][Expr] supports deserialization using the [serde] library to make flexible
//! configuration easy to set up, if the feature `serde` is enabled (disable by default).
//!
#![cfg_attr(feature = "serde", doc = " ```rust")]
#![cfg_attr(not(feature = "serde"), doc = " ```rust,ignore")]
//! #[macro_use]
//! extern crate serde_derive;
//! extern crate toml;
//! extern crate meval;
//! use meval::{Expr, Context};
//!
//! #[derive(Deserialize)]
//! struct Ode {
//!     #[serde(deserialize_with = "meval::de::as_f64")]
//!     x0: f64,
//!     #[serde(deserialize_with = "meval::de::as_f64")]
//!     t0: f64,
//!     f: Expr,
//!     g: Expr,
//! }
//!
//! fn main() {
//!     let config = r#"
//!         x0 = "cos(1.)"
//!         t0 = 2
//!         f = "sin(x)"
//!         g = 2.5
//!     "#;
//!     let ode: Ode = toml::from_str(config).unwrap();
//!
//!     assert_eq!(ode.x0, 1f64.cos());
//!     assert_eq!(ode.t0, 2f64);
//!     assert_eq!(ode.f.bind("x").unwrap()(2.), 2f64.sin());
//!     assert_eq!(ode.g.eval().unwrap(), 2.5f64);
//! }
//!
//! ```
//!
//! # Related projects
//!
//! This is a toy project of mine for learning Rust, and to be hopefully useful when writing
//! command line scripts. There is no plan to make this anything more than _math expression ->
//! number_ "converter". For more advanced scripting, see:
//!
//! - [dyon] -- A rusty dynamically typed scripting language
//! - [gluon] -- A static, type inferred programming language for application embedding
//! - [rodolf0/tox](https://github.com/rodolf0/tox) -- another shunting yard expression parser
//!
//! [Rust]: https://www.rust-lang.org/
//! [std-float]: http://doc.rust-lang.org/stable/std/primitive.f64.html
//!
//! [Expr]: struct.Expr.html
//! [Expr::bind]: struct.Expr.html#method.bind
//! [Context]: struct.Context.html
//! [serde]: https://crates.io/crates/serde
//! [dyon]: https://crates.io/crates/dyon
//! [gluon]: https://crates.io/crates/gluon

#[macro_use]
extern crate nom;
extern crate fnv;
#[cfg(feature = "serde")]
extern crate serde;
#[cfg_attr(all(test, feature = "serde"), macro_use)]
#[cfg(all(test, feature = "serde"))]
extern crate serde_derive;
#[cfg(test)]
extern crate serde_json;
#[cfg(test)]
extern crate serde_test;

use std::fmt;

mod expr;
pub mod shunting_yard;
pub mod tokenizer;

#[cfg(feature = "serde")]
pub mod de;

pub use expr::*;
pub use shunting_yard::RPNError;
pub use tokenizer::ParseError;

/// An error produced during parsing or evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    UnknownVariable(String),
    Function(String, FuncEvalError),
    /// An error returned by the parser.
    ParseError(ParseError),
    /// The shunting-yard algorithm returned an error.
    RPNError(RPNError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::UnknownVariable(ref name) => {
                write!(f, "Evaluation error: unknown variable `{}`.", name)
            }
            Error::Function(ref name, ref e) => {
                write!(f, "Evaluation error: function `{}`: {}", name, e)
            }
            Error::ParseError(ref e) => {
                try!(write!(f, "Parse error: "));
                e.fmt(f)
            }
            Error::RPNError(ref e) => {
                try!(write!(f, "RPN error: "));
                e.fmt(f)
            }
        }
    }
}

impl From<ParseError> for Error {
    fn from(err: ParseError) -> Error {
        Error::ParseError(err)
    }
}

impl From<RPNError> for Error {
    fn from(err: RPNError) -> Error {
        Error::RPNError(err)
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::UnknownVariable(_) => "unknown variable",
            Error::Function(_, _) => "function evaluation error",
            Error::ParseError(ref e) => e.description(),
            Error::RPNError(ref e) => e.description(),
        }
    }

    fn cause(&self) -> Option<&std::error::Error> {
        match *self {
            Error::ParseError(ref e) => Some(e),
            Error::RPNError(ref e) => Some(e),
            Error::Function(_, ref e) => Some(e),
            _ => None,
        }
    }
}
