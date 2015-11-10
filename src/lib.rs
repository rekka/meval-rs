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
    ParseError(ParseError),
    RPNError(RPNError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::UnknownVariable(ref name) =>
                write!(f, "Evaluation error: unknown variable `{}`", name),
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
