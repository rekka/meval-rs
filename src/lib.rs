#[macro_use]
extern crate nom;

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

