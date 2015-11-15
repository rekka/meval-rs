//! Tokenizer that converts a mathematical expression in a string form into a series of `Token`s.
//!
//! The underlying parser is build using the [nom] parser combinator crate.
//!
//! The parser should tokenize only well-formed expressions.
//!
//! [nom]: https://crates.io/crates/nom
use std::str::from_utf8;
use nom::{IResult, Needed, multispace, slice_to_offsets};

/// An error reported by the parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// A token that is not allowed at the given location (contains the location of the offending
    /// character in the source string).
    UnexpectedToken(usize),
    /// Missing right parentheses at the end of the source string (contains the number of missing
    /// parens).
    MissingRParen(i32),
    /// Missing operator or function argument at the end of the expression.
    MissingArgument,
    Unexpected,
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

    /// A number.
    Number(f64),
    /// A variable.
    Var(String),
    /// A function.
    Func(String),
}

named!(binop<Token>, alt!(
    chain!(tag!("+"),||{Token::Binary(Operation::Plus)}) |
    chain!(tag!("-"),||{Token::Binary(Operation::Minus)}) |
    chain!(tag!("*"),||{Token::Binary(Operation::Times)}) |
    chain!(tag!("/"),||{Token::Binary(Operation::Div)}) |
    chain!(tag!("^"),||{Token::Binary(Operation::Pow)})
    )
);

named!(unop<Token>, alt!(
    chain!(tag!("+"),||{Token::Unary(Operation::Plus)}) |
    chain!(tag!("-"),||{Token::Unary(Operation::Minus)})
    )
);

named!(lparen<Token>, chain!(tag!("("),||{Token::LParen}));
named!(rparen<Token>, chain!(tag!(")"),||{Token::RParen}));

/// Parse an identifier:
///
/// Must start with a letter or an underscore, can be followed by letters, digits or underscores.
fn ident(input: &[u8]) -> IResult<&[u8], &[u8]> {
    use nom::IResult::*;
    use nom::{Needed, ErrorKind};
    use nom::Err::*;

    // first character must be 'a'...'z' | 'A'...'Z' | '_'
    match input.first().map(|&c| c as char) {
        Some('a'...'z') | Some('A'...'Z') | Some('_') => {
            let n = input.iter()
                         .skip(1)
                         .take_while(|&&c| {
                             match c as char {
                                 'a'...'z' | 'A'...'Z' | '_' | '0'...'9' => true,
                                 _ => false,
                             }
                         })
                         .count();
            let (parsed, rest) = input.split_at(n + 1);
            Done(rest, parsed)
        }
        None => Incomplete(Needed::Size(1)),
        _ => Error(Position(ErrorKind::Custom(0), input)),
    }
}

named!(var<Token>, map!(map_res!(ident, from_utf8), |s: &str| Token::Var(s.into())));

/// Parse `func(`, returns `func`.
named!(func<Token>, map!(map_res!(
            terminated!(ident,
                        preceded!(opt!(multispace), tag!("("))), from_utf8),
            |s: &str| Token::Func(s.into())
            )
      );

/// Matches one or more digit characters `0`...`9`.
///
/// Fix of IMHO broken `nom::digit`.
fn digit_fixed(input: &[u8]) -> IResult<&[u8], &[u8]> {
    use nom::IResult::*;
    use nom::{Needed, is_digit, ErrorKind};
    use nom::Err::*;
    if input.is_empty() {
        return Incomplete(Needed::Size(1));
    }
    for (idx, item) in input.iter().enumerate() {
        if !is_digit(*item) {
            if idx == 0 {
                return Error(Position(ErrorKind::Digit, input));
            } else {
                return Done(&input[idx..], &input[0..idx]);
            }
        }
    }
    Done(b"", input)
}

named!(float<usize>, chain!(
        a: digit_fixed ~
        b: opt!(chain!(tag!(".") ~ d: opt!(digit_fixed),||{1 + d.map(|s| s.len()).unwrap_or(0)})) ~
        e: exp,
        ||{a.len() + b.unwrap_or(0) + e.unwrap_or(0)}
    )
);

/// Parser that matches the exponential part of a float. If the `input[0] == 'e' | 'E'` then at
/// least one digit must match.
fn exp(input: &[u8]) -> IResult<&[u8], Option<usize>> {
    use nom::IResult::*;
    match alt!(input, tag!("e") | tag!("E")) {
        Incomplete(_) => Done(input, None),
        Error(_) => Done(input, None),
        Done(i, _) => {
            match chain!(i, s: alt!(tag!("+") | tag!("-"))? ~
                   e: digit_fixed,
                ||{Some(1 + s.map(|s| s.len()).unwrap_or(0) + e.len())}) {
                Incomplete(Needed::Size(i)) => {
                    Incomplete(Needed::Size(i + 1))
                }
                o => o,
            }
        }
    }
}

fn number(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::IResult::*;
    use nom::Err;
    use std::str::FromStr;
    use nom::ErrorKind;

    match float(input) {
        Done(rest, l) => {
            // it should be safe to call unwrap here instead of the error checking, since
            // `float` should match only well-formed numbers
            from_utf8(&input[..l])
                .ok()
                .and_then(|s| f64::from_str(s).ok())
                .map_or(Error(Err::Position(ErrorKind::Custom(0), input)),
                        |f| Done(rest, Token::Number(f)))
        }
        Error(e) => Error(e),
        Incomplete(n) => Incomplete(n),
    }
}

named!(lexpr<Token>, delimited!(opt!(multispace),
                                alt!(number | complete!(func) | var | unop | lparen),
                                opt!(multispace)));
named!(after_rexpr<Token>, delimited!(opt!(multispace),
                                      alt!(binop | rparen),
                                      opt!(multispace)));
named!(after_rexpr_no_paren<Token>, delimited!(opt!(multispace),
                                               alt!(binop),
                                               opt!(multispace)));

#[derive(Debug, PartialEq, Eq)]
enum TokenizerState {
    // accept any token that is an expression from the left: var, num, (, unop
    LExpr,
    // accept any token that needs an expression on the left: binop, )
    AfterRExpr,
}

/// Tokenize a given mathematical expression.
///
/// The parser should return `Ok` only if the expression is well-formed.
///
/// # Failure
///
/// Returns `Err` if the expression is not well-formed.
pub fn tokenize<S: AsRef<str>>(input: S) -> Result<Vec<Token>, ParseError> {
    use self::TokenizerState::*;
    use nom::IResult::*;
    use nom::Err;
    let mut state = LExpr;
    // number of function arguments left
    let mut paren_stack = vec![];

    let mut res = vec![];

    let input = input.as_ref().as_bytes();
    let mut s = input;

    while !s.is_empty() {
        let r = match state {
            LExpr => lexpr(s),
            AfterRExpr if paren_stack.is_empty() => after_rexpr_no_paren(s),
            AfterRExpr => after_rexpr(s),
        };

        match r {
            Done(m, t) => {
                match t {
                    Token::LParen | Token::Func(_) => {
                        paren_stack.push(0);
                    }
                    Token::RParen => {
                        paren_stack.pop().expect("The paren_stack is empty!");
                    }
                    Token::Var(_) => {
                        state = AfterRExpr;
                    }
                    Token::Number(_) => {
                        state = AfterRExpr;
                    }
                    Token::Binary(_) => {
                        state = LExpr;
                    }
                    _ => {}
                }
                res.push(t);
                s = m;
            }
            Error(Err::Position(_, p)) => {
                let (i, _) = slice_to_offsets(input, p);
                return Err(ParseError::UnexpectedToken(i));
            }
            _ => {
                panic!("Unexpected parse result when parsing `{}` at `{}`: {:?}",
                       String::from_utf8_lossy(input),
                       String::from_utf8_lossy(s),
                       r);
            }
        }
    }

    match state {
        LExpr => Err(ParseError::MissingArgument),
        _ if !paren_stack.is_empty() => Err(ParseError::MissingRParen(paren_stack.len() as i32)),
        _ => Ok(res),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{number, binop, var, func};
    use nom::{IResult, Needed};
    use nom::ErrorKind::*;
    use nom::Err::*;

    #[test]
    fn it_works() {
        assert_eq!(binop(b"+"),
                   IResult::Done(&b""[..], Token::Binary(Operation::Plus)));
        assert_eq!(number(b"32143"),
                   IResult::Done(&b""[..], Token::Number(32143f64)));
        assert_eq!(var(b"abc"),
                   IResult::Done(&b""[..], Token::Var("abc".into())));
        assert_eq!(func(b"abc("),
                   IResult::Done(&b""[..], Token::Func("abc".into())));
        assert_eq!(func(b"abc ("),
                   IResult::Done(&b""[..], Token::Func("abc".into())));
    }

    #[test]
    fn test_var() {
        for &s in ["abc", "U0", "_034", "a_be45EA"].iter() {
            assert_eq!(var(s.as_bytes()),
                       IResult::Done(&b""[..], Token::Var(s.into())));
        }

        assert_eq!(var(b""), IResult::Incomplete(Needed::Size(1)));
        assert_eq!(var(b"0"), IResult::Error(Position(Custom(0), &b"0"[..])));
    }

    #[test]
    fn test_func() {
        for &s in ["abc(", "u0(", "_034 (", "A_be45EA  ("].iter() {
            assert_eq!(func(s.as_bytes()),
                       IResult::Done(&b""[..], Token::Func((&s[0..s.len() - 1]).trim().into())));
        }

        assert_eq!(func(b""), IResult::Incomplete(Needed::Size(1)));
        assert_eq!(func(b"("), IResult::Error(Position(Custom(0), &b"("[..])));
        assert_eq!(func(b"0("), IResult::Error(Position(Custom(0), &b"0("[..])));
    }

    #[test]
    fn test_number() {
        use nom::IResult::*;

        assert_eq!(number(b"32143"),
                   IResult::Done(&b""[..], Token::Number(32143f64)));
        assert_eq!(number(b"2."),
                   IResult::Done(&b""[..], Token::Number(2.0f64)));
        assert_eq!(number(b"32143.25"),
                   IResult::Done(&b""[..], Token::Number(32143.25f64)));
        assert_eq!(number(b"0.125e9"),
                   IResult::Done(&b""[..], Token::Number(0.125e9f64)));
        assert_eq!(number(b"20.5E-3"),
                   IResult::Done(&b""[..], Token::Number(20.5E-3f64)));
        assert_eq!(number(b"123423e+50"),
                   IResult::Done(&b""[..], Token::Number(123423e+50f64)));

        assert_eq!(number(b""), IResult::Incomplete(Needed::Size(1)));
        assert_eq!(number(b".2"), IResult::Error(Position(Digit, &b".2"[..])));
        assert_eq!(number(b"+"), IResult::Error(Position(Digit, &b"+"[..])));
        assert_eq!(number(b"e"), IResult::Error(Position(Digit, &b"e"[..])));
        assert_eq!(number(b"1E"), IResult::Incomplete(Needed::Size(3)));
        assert_eq!(number(b"1e+"), IResult::Incomplete(Needed::Size(4)));
    }

    #[test]
    fn test_tokenize() {
        use super::Token::*;
        use super::Operation::*;

        assert_eq!(tokenize("a"), Ok(vec![Var("a".into())]));

        assert_eq!(tokenize("2 +(3--2) "),
                   Ok(vec![Number(2f64),
                           Binary(Plus),
                           LParen,
                           Number(3f64),
                           Binary(Minus),
                           Unary(Minus),
                           Number(2f64),
                           RParen]));

        assert_eq!(tokenize("-2^ ab0 *12 - C_0"),
                   Ok(vec![Unary(Minus),
                           Number(2f64),
                           Binary(Pow),
                           Var("ab0".into()),
                           Binary(Times),
                           Number(12f64),
                           Binary(Minus),
                           Var("C_0".into()),
                   ]));

        assert_eq!(tokenize("-sin(pi * 3)^ cos(2) / Func2(x) * _buildIN(y)"),
                   Ok(vec![Unary(Minus),
                           Func("sin".into()),
                           Var("pi".into()),
                           Binary(Times),
                           Number(3f64),
                           RParen,
                           Binary(Pow),
                           Func("cos".into()),
                           Number(2f64),
                           RParen,
                           Binary(Div),
                           Func("Func2".into()),
                           Var("x".into()),
                           RParen,
                           Binary(Times),
                           Func("_buildIN".into()),
                           Var("y".into()),
                           RParen,
                       ]));

        assert_eq!(tokenize(""), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("2)"), Err(ParseError::UnexpectedToken(1)));
        assert_eq!(tokenize("2^"), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("(((2)"), Err(ParseError::MissingRParen(2)));
    }
}
