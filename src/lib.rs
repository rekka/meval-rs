//! This [Rust] crate provides a simple math expression parser. Its main
//! goal is to be convenient to use by default, while allowing for some
//! flexibility.
//!
//! For other similar projects see:
//!
//! - [rodolf0/tox](https://github.com/rodolf0/tox)
//!
//! ## Simple examples
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
//! Need to define a rust function from an expression? No problem:
//!
//! ```rust
//! extern crate meval;
//!
//! fn main() {
//!     let expr = meval::Expr::from_str("sin(pi * x)").unwrap();
//!     let func = expr.bind("x").unwrap();
//!
//!     let vs: Vec<_> = (0..100+1).map(|i| func(i as f64 / 100.)).collect();
//!
//!     println!("sin(pi * x), 0 <= x <= 1: {:?}", vs);
//! }
//! ```
//!
//! [Rust]: https://www.rust-lang.org/
//!
#[macro_use]
extern crate nom;

use std::fmt;

pub mod tokenizer;
pub mod shunting_yard;
mod expr;

pub use expr::*;
pub use shunting_yard::RPNError;
pub use tokenizer::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    UnknownVariable(String),
    UnknownFunction(String),
    ParseError(ParseError),
    RPNError(RPNError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::UnknownVariable(ref name) =>
                write!(f, "Evaluation error: unknown variable `{}`", name),
            Error::UnknownFunction(ref name) =>
                write!(f, "Evaluation error: unknown function `{}`", name),
            Error::ParseError(ref e) => write!(f, "Parse error: {:?}", e),
            Error::RPNError(ref e) => write!(f, "RPN error: {:?}", e),
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
