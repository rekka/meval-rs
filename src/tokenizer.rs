//! Tokenizer that converts a mathematical expression in a string form into a series of `Token`s.
//!
//! The underlying parser is build using the [nom] parser combinator crate.
//!
//! The parser should tokenize only well-formed expressions.
//!
//! [nom]: https://crates.io/crates/nom
use nom::{multispace, slice_to_offsets, IResult, Needed};
use std;
use std::fmt;
use std::str::from_utf8;

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
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ParseError::UnexpectedToken(i) => write!(f, "Unexpected token at byte {}.", i),
            ParseError::MissingRParen(i) => write!(
                f,
                "Missing {} right parenthes{}.",
                i,
                if i == 1 { "is" } else { "es" }
            ),
            ParseError::MissingArgument => write!(f, "Missing argument at the end of expression."),
        }
    }
}

impl std::error::Error for ParseError {
    fn description(&self) -> &str {
        match *self {
            ParseError::UnexpectedToken(_) => "unexpected token",
            ParseError::MissingRParen(_) => "missing right parenthesis",
            ParseError::MissingArgument => "missing argument",
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

    /// A number.
    Number(f64),
    /// A variable.
    Var(String),
    /// A function with name and number of arguments.
    Func(String, Option<usize>),
}

named!(
    binop<Token>,
    alt!(
        chain!(tag!("+"), || Token::Binary(Operation::Plus))
            | chain!(tag!("-"), || Token::Binary(Operation::Minus))
            | chain!(tag!("*"), || Token::Binary(Operation::Times))
            | chain!(tag!("/"), || Token::Binary(Operation::Div))
            | chain!(tag!("%"), || Token::Binary(Operation::Rem))
            | chain!(tag!("^"), || Token::Binary(Operation::Pow))
    )
);

named!(
    unop<Token>,
    alt!(
        chain!(tag!("+"), || Token::Unary(Operation::Plus))
            | chain!(tag!("-"), || Token::Unary(Operation::Minus))
    )
);

named!(lparen<Token>, chain!(tag!("("), || Token::LParen));
named!(rparen<Token>, chain!(tag!(")"), || Token::RParen));
named!(comma<Token>, chain!(tag!(","), || Token::Comma));

/// Parse an identifier:
///
/// Must start with a letter or an underscore, can be followed by letters, digits or underscores.
fn ident(input: &[u8]) -> IResult<&[u8], &[u8]> {
    use nom::Err::*;
    use nom::IResult::*;
    use nom::{ErrorKind, Needed};

    // first character must be 'a'...'z' | 'A'...'Z' | '_'
    match input.first().cloned() {
        Some(b'a'...b'z') | Some(b'A'...b'Z') | Some(b'_') => {
            let n = input
                .iter()
                .skip(1)
                .take_while(|&&c| match c {
                    b'a'...b'z' | b'A'...b'Z' | b'_' | b'0'...b'9' => true,
                    _ => false,
                }).count();
            let (parsed, rest) = input.split_at(n + 1);
            Done(rest, parsed)
        }
        None => Incomplete(Needed::Size(1)),
        _ => Error(Position(ErrorKind::Custom(0), input)),
    }
}

named!(
    var<Token>,
    map!(map_res!(complete!(ident), from_utf8), |s: &str| Token::Var(
        s.into()
    ))
);

/// Parse `func(`, returns `func`.
named!(
    func<Token>,
    map!(
        map_res!(
            terminated!(
                complete!(ident),
                preceded!(opt!(multispace), complete!(tag!("(")))
            ),
            from_utf8
        ),
        |s: &str| Token::Func(s.into(), None)
    )
);

/// Matches one or more digit characters `0`...`9`.
///
/// Never returns `nom::IResult::Incomplete`.
///
/// Fix of IMHO broken `nom::digit`, which parses an empty string successfully.
fn digit_complete(input: &[u8]) -> IResult<&[u8], &[u8]> {
    use nom::Err::*;
    use nom::IResult::*;
    use nom::{is_digit, ErrorKind};

    let n = input.iter().take_while(|&&c| is_digit(c)).count();
    if n > 0 {
        let (parsed, rest) = input.split_at(n);
        Done(rest, parsed)
    } else {
        Error(Position(ErrorKind::Digit, input))
    }
}

named!(
    float<usize>,
    chain!(
        a: digit_complete ~
        b: complete!(chain!(tag!(".") ~ d: digit_complete?,
                            ||{1 + d.map(|s| s.len()).unwrap_or(0)}))? ~
        e: complete!(exp),
        ||{a.len() + b.unwrap_or(0) + e.unwrap_or(0)}
    )
);

/// Parser that matches the exponential part of a float. If the `input[0] == 'e' | 'E'` then at
/// least one digit must match.
fn exp(input: &[u8]) -> IResult<&[u8], Option<usize>> {
    use nom::IResult::*;
    match alt!(input, tag!("e") | tag!("E")) {
        Incomplete(_) | Error(_) => Done(input, None),
        Done(i, _) => match chain!(i, s: alt!(tag!("+") | tag!("-"))? ~
                   e: digit_complete,
                ||{Some(1 + s.map(|s| s.len()).unwrap_or(0) + e.len())})
        {
            Incomplete(Needed::Size(i)) => Incomplete(Needed::Size(i + 1)),
            o => o,
        },
    }
}

fn number(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Err;
    use nom::ErrorKind;
    use nom::IResult::*;
    use std::str::FromStr;

    match float(input) {
        Done(rest, l) => {
            // it should be safe to call unwrap here instead of the error checking, since
            // `float` should match only well-formed numbers
            from_utf8(&input[..l])
                .ok()
                .and_then(|s| f64::from_str(s).ok())
                .map_or(Error(Err::Position(ErrorKind::Custom(0), input)), |f| {
                    Done(rest, Token::Number(f))
                })
        }
        Error(e) => Error(e),
        Incomplete(n) => Incomplete(n),
    }
}

named!(
    lexpr<Token>,
    delimited!(
        opt!(multispace),
        alt!(number | func | var | unop | lparen),
        opt!(multispace)
    )
);
named!(
    after_rexpr<Token>,
    delimited!(opt!(multispace), alt!(binop | rparen), opt!(multispace))
);
named!(
    after_rexpr_no_paren<Token>,
    delimited!(opt!(multispace), alt!(binop), opt!(multispace))
);
named!(
    after_rexpr_comma<Token>,
    delimited!(
        opt!(multispace),
        alt!(binop | rparen | comma),
        opt!(multispace)
    )
);

#[derive(Debug, Clone, Copy)]
enum TokenizerState {
    // accept any token that is an expression from the left: var, num, (, unop
    LExpr,
    // accept any token that needs an expression on the left: binop, ), comma
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
pub fn tokenize<S: AsRef<str>>(input: S) -> Result<Vec<Token>, ParseError> {
    use self::TokenizerState::*;
    use nom::Err;
    use nom::IResult::*;
    let mut state = LExpr;
    // number of function arguments left
    let mut paren_stack = vec![];

    let mut res = vec![];

    let input = input.as_ref().as_bytes();
    let mut s = input;

    while !s.is_empty() {
        let r = match (state, paren_stack.last()) {
            (LExpr, _) => lexpr(s),
            (AfterRExpr, None) => after_rexpr_no_paren(s),
            (AfterRExpr, Some(&ParenState::Subexpr)) => after_rexpr(s),
            (AfterRExpr, Some(&ParenState::Func)) => after_rexpr_comma(s),
        };

        match r {
            Done(rest, t) => {
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
                        state = AfterRExpr;
                    }
                    Token::Binary(_) | Token::Comma => {
                        state = LExpr;
                    }
                    _ => {}
                }
                res.push(t);
                s = rest;
            }
            Error(Err::Position(_, p)) => {
                let (i, _) = slice_to_offsets(input, p);
                return Err(ParseError::UnexpectedToken(i));
            }
            _ => {
                panic!(
                    "Unexpected parse result when parsing `{}` at `{}`: {:?}",
                    String::from_utf8_lossy(input),
                    String::from_utf8_lossy(s),
                    r
                );
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
    use super::{binop, func, number, var};
    use nom::Err::*;
    use nom::ErrorKind::*;
    use nom::IResult;

    #[test]
    fn it_works() {
        assert_eq!(
            binop(b"+"),
            IResult::Done(&b""[..], Token::Binary(Operation::Plus))
        );
        assert_eq!(
            number(b"32143"),
            IResult::Done(&b""[..], Token::Number(32143f64))
        );
        assert_eq!(
            var(b"abc"),
            IResult::Done(&b""[..], Token::Var("abc".into()))
        );
        assert_eq!(
            func(b"abc("),
            IResult::Done(&b""[..], Token::Func("abc".into(), None))
        );
        assert_eq!(
            func(b"abc ("),
            IResult::Done(&b""[..], Token::Func("abc".into(), None))
        );
    }

    #[test]
    fn test_var() {
        for &s in ["abc", "U0", "_034", "a_be45EA", "aAzZ_"].iter() {
            assert_eq!(
                var(s.as_bytes()),
                IResult::Done(&b""[..], Token::Var(s.into()))
            );
        }

        assert_eq!(var(b""), IResult::Error(Position(Complete, &b""[..])));
        assert_eq!(var(b"0"), IResult::Error(Position(Custom(0), &b"0"[..])));
    }

    #[test]
    fn test_func() {
        for &s in ["abc(", "u0(", "_034 (", "A_be45EA  ("].iter() {
            assert_eq!(
                func(s.as_bytes()),
                IResult::Done(
                    &b""[..],
                    Token::Func((&s[0..s.len() - 1]).trim().into(), None)
                )
            );
        }

        assert_eq!(func(b""), IResult::Error(Position(Complete, &b""[..])));
        assert_eq!(func(b"("), IResult::Error(Position(Custom(0), &b"("[..])));
        assert_eq!(func(b"0("), IResult::Error(Position(Custom(0), &b"0("[..])));
    }

    #[test]
    fn test_number() {
        assert_eq!(
            number(b"32143"),
            IResult::Done(&b""[..], Token::Number(32143f64))
        );
        assert_eq!(
            number(b"2."),
            IResult::Done(&b""[..], Token::Number(2.0f64))
        );
        assert_eq!(
            number(b"32143.25"),
            IResult::Done(&b""[..], Token::Number(32143.25f64))
        );
        assert_eq!(
            number(b"0.125e9"),
            IResult::Done(&b""[..], Token::Number(0.125e9f64))
        );
        assert_eq!(
            number(b"20.5E-3"),
            IResult::Done(&b""[..], Token::Number(20.5E-3f64))
        );
        assert_eq!(
            number(b"123423e+50"),
            IResult::Done(&b""[..], Token::Number(123423e+50f64))
        );

        assert_eq!(number(b""), IResult::Error(Position(Digit, &b""[..])));
        assert_eq!(number(b".2"), IResult::Error(Position(Digit, &b".2"[..])));
        assert_eq!(number(b"+"), IResult::Error(Position(Digit, &b"+"[..])));
        assert_eq!(number(b"e"), IResult::Error(Position(Digit, &b"e"[..])));
        assert_eq!(number(b"1E"), IResult::Error(Position(Complete, &b"E"[..])));
        assert_eq!(number(b"1e+"), IResult::Error(Position(Digit, &b""[..])));
    }

    #[test]
    fn test_tokenize() {
        use super::Operation::*;
        use super::Token::*;

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

        assert_eq!(tokenize("()"), Err(ParseError::UnexpectedToken(1)));

        assert_eq!(tokenize(""), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("2)"), Err(ParseError::UnexpectedToken(1)));
        assert_eq!(tokenize("2^"), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("(((2)"), Err(ParseError::MissingRParen(2)));
        assert_eq!(tokenize("f(2,)"), Err(ParseError::UnexpectedToken(4)));
        assert_eq!(tokenize("f(,2)"), Err(ParseError::UnexpectedToken(2)));
    }
}
