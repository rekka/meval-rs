#[macro_use]
extern crate nom;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Operation {
    Plus,
    Minus,
    Times,
    Div,
    Rem,
    Pow,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Binary(Operation),
    Unary(Operation),

    LParen,
    RParen,

    Number(f64),
    Var(String),
}

pub mod tokenizer;
pub mod shunting_yard;
mod expr;

pub use expr::*;
pub use shunting_yard::RPNError;
pub use tokenizer::ParseError;
