#[macro_use]
extern crate nom;

#[derive(Debug, PartialEq)]
pub enum Operation {
    Plus,
    Minus,
    Times,
    Div,
    Rem,
    Pow,
}

#[derive(Debug, PartialEq)]
pub enum Token {
    Binary(Operation),
    Unary(Operation),

    LParen,
    RParen,

    Number(f64),
    Var(String),

    Unknown,
}

pub mod tokenizer;
