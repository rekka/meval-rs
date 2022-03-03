//! Tokenizer that converts a mathematical expression in a string form into a series of `Token`s.
//!
//! The underlying parser is build using the [nom] parser combinator crate.
//!
//! The parser should tokenize only well-formed expressions.
//!
//! [nom]: https://crates.io/crates/nom

use nom::{
    branch::alt,
    bytes::complete::is_a,
    bytes::complete::tag,
    character::complete::{multispace0, one_of},
    combinator::{complete, map, opt, peek, recognize},
    error::ErrorKind,
    number::complete::double,
    sequence::{delimited, preceded, terminated},
    Err, IResult,
};

use std::fmt;

use std::f64;

/// An error reported by the parser.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenParseError {
    /// A token that is not allowed at the given location (contains the location of the offending
    /// character in the source string).
    UnexpectedToken(usize),

    UnexpectedStrToken(String),
    /// Missing right parentheses at the end of the source string (contains the number of missing
    /// parens).
    MissingRParen(i32),
    /// Missing operator or function argument at the end of the expression.
    MissingArgument,

    UnknownError,
}

impl fmt::Display for TokenParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &*self {
            TokenParseError::UnexpectedToken(i) => write!(f, "Unexpected token at byte {}.", i),
            TokenParseError::UnexpectedStrToken(s) => write!(f, "Unexpected token {}.", s),
            TokenParseError::MissingRParen(i) => write!(
                f,
                "Missing {} right parenthes{}.",
                i,
                if *i == 1 { "is" } else { "es" }
            ),
            TokenParseError::MissingArgument => {
                write!(f, "Missing argument at the end of expression.")
            }
            TokenParseError::UnknownError => write!(f, "Unknown pass error."),
        }
    }
}

impl std::error::Error for TokenParseError {
    fn description(&self) -> &str {
        match *self {
            TokenParseError::UnexpectedToken(_) => "unexpected token",
            TokenParseError::UnexpectedStrToken(_) => "Unexpected token",
            TokenParseError::MissingRParen(_) => "missing right parenthesis",
            TokenParseError::MissingArgument => "missing argument",
            TokenParseError::UnknownError => "unknown error",
        }
    }
}

/// Mathematical operations.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Operation {
    Plus,
    Minus,
    Times,
    Div,
    Rem,
    Pow,
    Fact,
}

/// Expression tokens.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    /// Binary operation.
    Binary(Operation),
    /// Unary operation.
    Unary(Operation),

    /// Left parenthesis.
    LParen,
    /// Right parenthesis.
    RParen,
    /// Comma: function argument separator
    Comma,
    /// Decimal Point
    //DecimalPoint,

    /// A number.
    Number(f64),
    /// A variable.
    Var(String),
    /// A function with name and number of arguments.
    Func(String, Option<usize>),
}

/// Continuing the trend of starting from the simplest piece and building up,
/// we start by creating a parser for the built-in operator functions.
fn binop<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    // one_of matches one of the characters we give it
    let (i, t) = one_of("+-*/%^!")(i)?;

    // because we are matching single character tokens, we can do the matching logic
    // on the returned value
    Ok((
        i,
        match t {
            '+' => Token::Binary(Operation::Plus),
            '-' => Token::Binary(Operation::Minus),
            '*' => Token::Binary(Operation::Times),
            '/' => Token::Binary(Operation::Div),
            '%' => Token::Binary(Operation::Rem),
            '^' => Token::Binary(Operation::Pow),
            '!' => Token::Binary(Operation::Fact),
            _ => unreachable!(),
        },
    ))
}

fn lparen<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    map(tag("("), |_: &str| Token::LParen)(i)
}

fn rparen<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    map(tag(")"), |_: &str| Token::RParen)(i)
}

fn comma<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    map(tag(","), |_: &str| Token::Comma)(i)
}

/// negpos parse. detects either - or +
fn negpos_s<'a>(i: &'a str) -> IResult<&'a str, &'a str, (&'a str, ErrorKind)> {
    match alt((tag("+"), tag("-")))(i) {
        Ok((remaining_input, operator)) => Ok((remaining_input, operator)),
        Err(e) => Err(e),
    }
}

/// negpos parse. detects either - or +
fn negpos<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    match negpos_s(i) {
        Ok((remaining_input, operator)) => match operator {
            "+" => Ok((remaining_input, Token::Unary(Operation::Plus))),
            "-" => Ok((remaining_input, Token::Unary(Operation::Minus))),
            _ => panic!("Should never occur"),
        },
        Err(e) => Err(e),
    }
}

/// factorial parse
fn fact<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    map(tag("!"), |_s: &str| Token::Unary(Operation::Fact))(i)
}

fn ident<'a>(i: &'a str) -> IResult<&'a str, &'a str, (&'a str, ErrorKind)> {
    let remaining_chars: &str = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let first_chars: &str = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // Returns whole strings matched by the given parser.
    recognize(
        // Runs the first parser, if succeeded then runs second, and returns the second result.
        // Note that returned ok value of `preceded()` is ignored by `recognize()`.
        preceded(
            // Parses a single character contained in the given string.
            one_of(first_chars),
            // Parses the longest slice consisting of the given characters
            opt(is_a(remaining_chars)),
        ),
    )(i)
}

fn var<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    map(complete(ident), |s: &str| Token::Var(s.into()))(i)
}

/// Parse `func(`, returns `func`.
fn func<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    map(
        //recognize(
        terminated(complete(ident), preceded(multispace0, complete(tag("(")))), //)
        |s: &str| Token::Func(s.into(), None),
    )(i)
}

fn number<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    preceded(peek(one_of("0123456789")), map(double, Token::Number))(i)
}

fn lexpr<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    delimited(
        multispace0,
        alt((number, func, var, negpos, lparen)),
        multispace0,
    )(i)
}

fn after_rexpr<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    delimited(multispace0, alt((fact, binop, rparen)), multispace0)(i)
}

fn after_rexpr_no_paren<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    delimited(multispace0, alt((fact, binop)), multispace0)(i)
}

fn after_rexpr_comma<'a>(i: &'a str) -> IResult<&'a str, Token, (&'a str, ErrorKind)> {
    delimited(multispace0, alt((fact, binop, rparen, comma)), multispace0)(i)
}

#[derive(Debug, Clone, Copy)]
enum TokenizerState {
    // accept any token that is an expression from the left: var, num, (, negpos
    LExpr,
    // accept any token that needs an expression on the left: fact, binop, ), comma
    AfterRExpr,
}

#[derive(Debug, Clone, Copy)]
enum ParenState {
    Subexpr,
    Func,
}

/// Tokenize a given mathematical expression.
///
/// The parser should return `Ok` only if the expression is well-formed.
///
/// # Failure
///
/// Returns `Err` if the expression is not well-formed.
pub fn tokenize(input: &str) -> Result<Vec<Token>, TokenParseError> {
    let mut state: TokenizerState = TokenizerState::LExpr;
    // number of function arguments left
    let mut paren_stack = vec![];

    let mut res = vec![];

    let mut s = input;

    while !s.is_empty() {
        let r = match (state, paren_stack.last()) {
            (TokenizerState::LExpr, _) => lexpr(s),
            (TokenizerState::AfterRExpr, None) => after_rexpr_no_paren(s),
            (TokenizerState::AfterRExpr, Some(&ParenState::Subexpr)) => after_rexpr(s),
            (TokenizerState::AfterRExpr, Some(&ParenState::Func)) => after_rexpr_comma(s),
        };

        match r {
            Ok((rest, t)) => {
                match t {
                    Token::LParen => {
                        paren_stack.push(ParenState::Subexpr);
                    }
                    Token::Func(..) => {
                        paren_stack.push(ParenState::Func);
                    }
                    Token::RParen => {
                        paren_stack.pop().expect("The paren_stack is empty!");
                    }
                    Token::Var(_) | Token::Number(_) => {
                        state = TokenizerState::AfterRExpr;
                    }
                    Token::Binary(_) | Token::Comma => {
                        state = TokenizerState::LExpr;
                    }
                    _ => {}
                }
                res.push(t);
                s = rest;
            }
            Err(e) => {
                match e {
                    Err::Error((value, _)) => {
                        return Err(TokenParseError::UnexpectedStrToken(value.to_string()));
                    }
                    _ => (),
                }

                return Err(TokenParseError::UnknownError);
            }
            // Error(Err::Position(_, p)) => {
            //     let (i, _) = slice_to_offsets(input, p);
            //     return Err(TokenParseError::UnexpectedToken(i));
            // }
            _ => {
                panic!(
                    "Unexpected parse result when parsing `{}` at `{}`: {:?}",
                    input, s, r
                );
            }
        }
    }

    match state {
        TokenizerState::LExpr => Err(TokenParseError::MissingArgument),

        _ => {
            if !paren_stack.is_empty() {
                return Err(TokenParseError::MissingRParen(paren_stack.len() as i32));
            }

            Ok(res)
        }
    }
}

// ok rest ["+(3--2) "]   t Number(2.0)
// ok rest ["(3--2) "]   t Binary(Plus)
// ok rest [51, 45, 45, 50, 41, 32]   t LParen
// ok rest [45, 45, 50, 41, 32]   t Number(3.0)
// ok rest [45, 50, 41, 32]   t Binary(Minus)
// ok rest [50, 41, 32]   t Unary(Minus)
// ok rest [") "]   t Number(2.0)
// ok rest []   t RParen
// state: AfterRExpr
// paren_stack: []
// Ok([Number(2.0), Binary(Plus), LParen, Number(3.0), Binary(Minus), Unary(Minus), Number(2.0), RParen])

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(binop("+"), Ok(("", Token::Binary(Operation::Plus))));
        assert_eq!(ident("abc32"), Ok(("", "abc32")));
        assert_eq!(func("abc("), Ok(("", Token::Func("abc".into(), None))));
        assert_eq!(func("abc ("), Ok(("", Token::Func("abc".into(), None))));
        assert_eq!(var("abc"), Ok(("", Token::Var("abc".into()))));
        assert_eq!(fact("!"), Ok(("", Token::Unary(Operation::Fact))));
        assert_eq!(negpos_s("+"), Ok(("", "+")));
        assert_eq!(negpos_s("-"), Ok(("", "-")));
        assert_eq!(negpos_s("+362"), Ok(("362", "+")));
        assert_eq!(negpos_s("-5734"), Ok(("5734", "-")));
        assert_eq!(negpos("+"), Ok(("", Token::Unary(Operation::Plus))));
        assert_eq!(negpos("-"), Ok(("", Token::Unary(Operation::Minus))));
        assert_eq!(negpos("+642"), Ok(("642", Token::Unary(Operation::Plus))));
        assert_eq!(negpos("-563"), Ok(("563", Token::Unary(Operation::Minus))));
        assert_eq!(lparen("("), Ok(("", Token::LParen)));
        assert_eq!(rparen(")"), Ok(("", Token::RParen)));
        assert_eq!(comma(","), Ok(("", Token::Comma)));
        assert_eq!(comma(","), Ok(("", Token::Comma)));
        assert_eq!(number("+1.34e2"), Ok(("", Token::Number(134.0))));
        assert_eq!(number("+1.34e+2"), Ok(("", Token::Number(134.0))));
        assert_eq!(number("3E+2"), Ok(("", Token::Number(300.0))));
        assert_eq!(number("+4E+2"), Ok(("", Token::Number(400.0))));
        assert_eq!(number("-4.76E+2"), Ok(("", Token::Number(-476.0))));
        assert_eq!(number("-4.76"), Ok(("", Token::Number(-4.76))));
        assert_eq!(number("+4.76"), Ok(("", Token::Number(4.76))));
        assert_eq!(number("1.1"), Ok(("", Token::Number(1.1))));
        assert_eq!(number("-1.1"), Ok(("", Token::Number(-1.1))));
        assert_eq!(number("123E-02"), Ok(("", Token::Number(1.23))));
        assert_eq!(number("+123E-02"), Ok(("", Token::Number(1.23))));
        assert_eq!(number("-123E-02"), Ok(("", Token::Number(-1.23))));
        assert_eq!(
            number("abc"),
            Err(Err::Error(("abc", nom::error::ErrorKind::Float)))
        );
    }

    #[test]
    fn test_lexpr() {
        // number, func, var, negpos, lparen
        assert_eq!(
            number("a"),
            Err(Err::Error(("a", nom::error::ErrorKind::Float)))
        );

        assert_eq!(func("a"), Err(Err::Error(("", nom::error::ErrorKind::Tag))));

        assert_eq!(var("a"), Ok(("", Token::Var("a".into()))));

        assert_eq!(lexpr("a"), Ok(("", Token::Var("a".into()))));

        assert_eq!(lexpr("2+"), Ok(("+", Token::Number(2.0))));

        assert_eq!(lexpr("2 +(3--2) "), Ok(("+(3--2) ", Token::Number(2.0))));

        println!("{:?}", number("+(3--2) "));

        assert_eq!(
            lexpr("+(3--2) "),
            Ok(("+(3--2) ", Token::Binary(Operation::Plus)))
        );
    }

    #[test]
    fn test_var() {
        for &s in ["abc", "U0", "_034", "a_be45EA", "aAzZ_"].iter() {
            assert_eq!(var(s), Ok(("", Token::Var(s.into()))));
        }

        assert_eq!(var(""), Err(Err::Error(("", nom::error::ErrorKind::OneOf))));
        assert_eq!(
            var("0"),
            Err(Err::Error(("0", nom::error::ErrorKind::OneOf)))
        );
    }

    #[test]
    fn test_number() {
        assert_eq!(number("45"), Ok(("", Token::Number(45.0))));

        assert_eq!(
            number("+(3--2) "),
            Err(Err::Error(("+(3--2) ", nom::error::ErrorKind::OneOf)))
        );

        assert_eq!(
            number("+3 "),
            Err(Err::Error(("+3 ", nom::error::ErrorKind::OneOf)))
        );

        assert_eq!(
            number("(3--2) "),
            Err(Err::Error(("(3--2) ", nom::error::ErrorKind::OneOf)))
        );

        assert_eq!(
            number("(3) "),
            Err(Err::Error(("(3) ", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(
            number("(3) - (2) "),
            Err(Err::Error(("(3) - (2) ", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(number("32143"), Ok(("", Token::Number(32143f64))));
        assert_eq!(number("2."), Ok(("", Token::Number(2.0f64))));
        assert_eq!(number("32143.25"), Ok(("", Token::Number(32143.25f64))));
        assert_eq!(number("0.125e9"), Ok(("", Token::Number(0.125e9f64))));
        assert_eq!(number("20.5E-3"), Ok(("", Token::Number(20.5E-3f64))));
        assert_eq!(number("123423e+50"), Ok(("", Token::Number(123423e+50f64))));
        assert_eq!(number("0.2"), Ok(("", Token::Number(0.2))));
        assert_eq!(
            number(""),
            Err(Err::Error(("", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(
            number("+"),
            Err(Err::Error(("+", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(
            number("e"),
            Err(Err::Error(("e", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(
            number("1E"),
            Err(Err::Error(("E", nom::error::ErrorKind::Eof)))
        );
        assert_eq!(
            number("1e"),
            Err(Err::Error(("e", nom::error::ErrorKind::Eof)))
        );
        assert_eq!(
            number("1e+"),
            Err(Err::Error(("+", nom::error::ErrorKind::Eof)))
        );
        assert_eq!(
            number("1e+-?%"),
            Err(Err::Error(("-?%", nom::error::ErrorKind::Eof)))
        );
        assert_eq!(
            number("2+"),
            Err(Err::Error(("+", nom::error::ErrorKind::Eof)))
        );
    }

    #[test]
    fn test_func() {
        for &s in ["abc(", "u0(", "_034 (", "A_be45EA  ("].iter() {
            assert_eq!(
                func(s),
                Ok(("", Token::Func(s[0..s.len() - 1].trim().into(), None)))
            );
        }

        assert_eq!(
            func(""),
            Err(Err::Error(("", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(
            func("("),
            Err(Err::Error(("(", nom::error::ErrorKind::OneOf)))
        );
        assert_eq!(
            func("0("),
            Err(Err::Error(("0(", nom::error::ErrorKind::OneOf)))
        );
    }

    #[test]
    fn test_tokenize() {
        use super::Operation::*;
        use super::Token::*;

        // Ok([Number(2.0), Binary(Plus), Number(2.0), Binary(Div), Number(3.0), Binary(Minus), Number(56.0), Binary(Plus), Func("sin", None), Number(3.0), RParen])
        // Ok([Number(2.0), Binary(Plus), Number(2.0), Binary(Div), Number(3.0), Binary(Minus), Number(56.0), Binary(Plus), Func("sin", None), Number(3.0), RParen])
        println!("{:?}", tokenize("2 + 2/3-56 + sin(3)"));

        assert_eq!(tokenize("a"), Ok(vec![Var("a".into())]));

        assert_eq!(
            tokenize("2 +(3--2) "),
            Ok(vec![
                Number(2f64),
                Binary(Plus),
                LParen,
                Number(3f64),
                Binary(Minus),
                Unary(Minus),
                Number(2f64),
                RParen
            ])
        );

        assert_eq!(
            tokenize("-2^ ab0 *12 - C_0"),
            Ok(vec![
                Unary(Minus),
                Number(2f64),
                Binary(Pow),
                Var("ab0".into()),
                Binary(Times),
                Number(12f64),
                Binary(Minus),
                Var("C_0".into()),
            ])
        );

        assert_eq!(
            tokenize("-sin(pi * 3)^ cos(2) / Func2(x, f(y), z) * _buildIN(y)"),
            Ok(vec![
                Unary(Minus),
                Func("sin".into(), None),
                Var("pi".into()),
                Binary(Times),
                Number(3f64),
                RParen,
                Binary(Pow),
                Func("cos".into(), None),
                Number(2f64),
                RParen,
                Binary(Div),
                Func("Func2".into(), None),
                Var("x".into()),
                Comma,
                Func("f".into(), None),
                Var("y".into()),
                RParen,
                Comma,
                Var("z".into()),
                RParen,
                Binary(Times),
                Func("_buildIN".into(), None),
                Var("y".into()),
                RParen,
            ])
        );

        assert_eq!(
            tokenize("2 % 3"),
            Ok(vec![Number(2f64), Binary(Rem), Number(3f64)])
        );

        assert_eq!(
            tokenize("1 + 3! + 1"),
            Ok(vec![
                Number(1f64),
                Binary(Plus),
                Number(3f64),
                Unary(Fact),
                Binary(Plus),
                Number(1f64)
            ])
        );

        assert_eq!(
            tokenize("!3"),
            Err(TokenParseError::UnexpectedStrToken("!3".to_string()))
        );

        assert_eq!(
            tokenize("()"),
            Err(TokenParseError::UnexpectedStrToken(")".to_string()))
        );

        assert_eq!(tokenize(""), Err(TokenParseError::MissingArgument));
        assert_eq!(
            tokenize("2)"),
            Err(TokenParseError::UnexpectedStrToken(")".to_string()))
        );
        assert_eq!(tokenize("2^"), Err(TokenParseError::MissingArgument));
        assert_eq!(tokenize("(((2)"), Err(TokenParseError::MissingRParen(2)));
        assert_eq!(
            tokenize("f(2,)"),
            Err(TokenParseError::UnexpectedStrToken(")".to_string()))
        );
        assert_eq!(
            tokenize("f(,2)"),
            Err(TokenParseError::UnexpectedStrToken(",2)".to_string()))
        );
    }
}
