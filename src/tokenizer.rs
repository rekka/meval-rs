use std::str::from_utf8;
use nom::{IResult, Needed, alpha, multispace, slice_to_offsets};

use Token;
use Operation;

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken(usize),
    MissingRParen(i32),
    MissingArgument,
    Unexpected,
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

named!(var<Token>, map!(map_res!(alpha, from_utf8), |s: &str| Token::Var(s.into())));

/// Matches one or more digit characters `0`...`9`.
///
/// Fix of IMHO broken `nom::digit`.
pub fn digit_fixed(input: &[u8]) -> IResult<&[u8], &[u8]> {
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
                                alt!(number | var | unop | lparen),
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
pub fn tokenize<S: AsRef<str>>(input: S) -> Result<Vec<Token>, ParseError> {
    use self::TokenizerState::*;
    use nom::IResult::*;
    use nom::Err;
    let mut state = (LExpr, 0);

    let mut res = vec![];

    let input = input.as_ref().as_bytes();
    let mut s = input;

    while !s.is_empty() {
        let r = match state {
            (LExpr, _) => lexpr(s),
            (AfterRExpr, 0) => after_rexpr_no_paren(s),
            (AfterRExpr, _) => after_rexpr(s),
        };

        match r {
            Done(m, t) => {
                match t {
                    Token::LParen => {
                        state.1 += 1;
                    }
                    Token::RParen => {
                        state.1 -= 1;
                    }
                    Token::Var(_) => {
                        state.0 = AfterRExpr;
                    }
                    Token::Number(_) => {
                        state.0 = AfterRExpr;
                    }
                    Token::Binary(_) => {
                        state.0 = LExpr;
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
                return Err(ParseError::Unexpected);
            }
        }
    }

    match state {
        (LExpr, _) => Err(ParseError::MissingArgument),
        (_, n_parens) if n_parens > 0 => Err(ParseError::MissingRParen(n_parens)),
        _ => Ok(res),
    }
}

#[cfg(test)]
mod tests {
    use {Operation, Token};
    use super::*;
    use super::{number, float, binop, var};
    use nom::{IResult, Needed};

    #[test]
    fn it_works() {
        println!("{:?}", float(b"32143"));
        assert_eq!(binop(b"+"),
                   IResult::Done(&b""[..], Token::Binary(Operation::Plus)));
        assert_eq!(number(b"32143"),
                   IResult::Done(&b""[..], Token::Number(32143f64)));
        assert_eq!(var(b"abc"),
                   IResult::Done(&b""[..], Token::Var("abc".into())));
    }

    #[test]
    fn test_number() {
        use Token::*;
        use nom::IResult::*;
        use nom::ErrorKind::*;
        use nom::Err::*;

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
        use Token::*;
        use Operation::*;

        assert_eq!(tokenize("2 +(3--2) "),
                   Ok(vec![Number(2f64),
                           Binary(Plus),
                           LParen,
                           Number(3f64),
                           Binary(Minus),
                           Unary(Minus),
                           Number(2f64),
                           RParen]));

        assert_eq!(tokenize("-2^ abc *12"),
                   Ok(vec![Unary(Minus),
                           Number(2f64),
                           Binary(Pow),
                           Var("abc".into()),
                           Binary(Times),
                           Number(12f64)]));

        assert_eq!(tokenize("2)"), Err(ParseError::UnexpectedToken(1)));
        assert_eq!(tokenize("2^"), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("(((2)"), Err(ParseError::MissingRParen(2)));
    }
}
