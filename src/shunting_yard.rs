//! Implementation of the shunting-yard algorithm for converting an infix expression to an
//! expression in reverse Polish notation (RPN).
//!
//! See the Wikipedia articles on the [shunting-yard algorithm][shunting] and on [reverse Polish
//! notation][RPN] for more details.
//!
//! [RPN]: https://en.wikipedia.org/wiki/Reverse_Polish_notation
//! [shunting]: https://en.wikipedia.org/wiki/Shunting-yard_algorithm
use tokenizer::Token;

#[derive(Debug, Clone, Copy)]
enum Associativity {
    Left,
    Right,
    NA,
}

/// An error produced by the shunting-yard algorightm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RPNError {
    /// An extra left parenthesis was found.
    MismatchedLParen(usize),
    /// An extra right parenthesis was found.
    MismatchedRParen(usize),
    /// Too few operands for some operator.
    NotEnoughOperands(usize),
    /// Too many operands reported.
    TooManyOperands,
}

/// Returns the operator precedence and associativity for a given token.
fn prec_assoc(token: &Token) -> (u32, Associativity) {
    use self::Associativity::*;
    use tokenizer::Token::*;
    use tokenizer::Operation::*;
    match *token {
        Binary(op) => {
            match op {
                Plus | Minus => (1, Left),
                Times | Div | Rem => (2, Left),
                Pow => (4, Right),
            }
        }
        Unary(op) => {
            match op {
                Plus | Minus => (3, NA),
                _ => unimplemented!(),
            }
        }
        Var(_) | Number(_) | Func(_) | LParen | RParen => (0, NA),
    }
}

/// Converts a tokenized infix expression to reverse Polish notation.
///
/// # Failure
///
/// Returns `Err` if the input expression is not well-formed.
pub fn to_rpn(input: &[Token]) -> Result<Vec<Token>, RPNError> {
    use tokenizer::Token::*;

    let mut output = Vec::with_capacity(input.len());
    let mut stack = Vec::with_capacity(input.len());

    for (index, token) in input.iter().enumerate() {
        let token = token.clone();
        match token {
            Number(_) | Var(_) => output.push(token),
            Unary(_) => stack.push((index, token)),
            Binary(_) => {
                let pa1 = prec_assoc(&token);
                while !stack.is_empty() {
                    let pa2 = prec_assoc(&stack.last().unwrap().1);
                    match (pa1, pa2) {
                        ((i, Associativity::Left), (j, _)) if i <= j => {
                            output.push(stack.pop().unwrap().1);
                        }
                        ((i, Associativity::Right), (j, _)) if i < j => {
                            output.push(stack.pop().unwrap().1);
                        }
                        _ => {
                            break;
                        }
                    }
                }
                stack.push((index, token))
            }
            LParen => stack.push((index, token)),
            RParen => {
                let mut found = false;
                while let Some((_, t)) = stack.pop() {
                    match t {
                        LParen => {
                            found = true;
                            break;
                        }
                        Func(_) => {
                            found = true;
                            output.push(t);
                            break;
                        }
                        _ => output.push(t),
                    }
                }
                if !found {
                    return Err(RPNError::MismatchedRParen(index));
                }
            }
            Func(_) => stack.push((index, token)),
        }
    }

    while let Some((index, token)) = stack.pop() {
        match token {
            Unary(_) | Binary(_) => output.push(token),
            LParen | Func(_) => return Err(RPNError::MismatchedLParen(index)),
            _ => panic!("Unexpected token on stack."),
        }
    }

    // verify rpn
    let mut n_operands = 0;
    for (index, token) in output.iter().enumerate() {
        match *token {
            Var(_) | Number(_) => n_operands += 1,
            Unary(_) => (),
            Binary(_) => n_operands -= 1,
            Func(_) => n_operands -= 1 - 1,
            _ => panic!("Nothing else should be here"),
        }
        if n_operands <= 0 {
            return Err(RPNError::NotEnoughOperands(index));
        }
    }

    if n_operands > 1 {
        return Err(RPNError::TooManyOperands);
    }


    output.shrink_to_fit();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizer::Token::*;
    use tokenizer::Operation::*;

    #[test]
    fn test_to_rpn() {
        assert_eq!(to_rpn(&[Number(1.)]), Ok(vec![Number(1.)]));
        assert_eq!(to_rpn(&[Number(1.), Binary(Plus), Number(2.)]),
                   Ok(vec![Number(1.), Number(2.), Binary(Plus)]));
        assert_eq!(to_rpn(&[Unary(Minus), Number(1.), Binary(Pow), Number(2.)]),
                   Ok(vec![Number(1.), Number(2.), Binary(Pow), Unary(Minus)]));
        assert_eq!(to_rpn(&[Number(3.), Binary(Minus), Number(1.), Binary(Times), Number(2.)]),
                   Ok(vec![Number(3.), Number(1.), Number(2.), Binary(Times), Binary(Minus)]));
        assert_eq!(to_rpn(&[LParen,
                            Number(3.),
                            Binary(Minus),
                            Number(1.),
                            RParen,
                            Binary(Times),
                            Number(2.)]),
                   Ok(vec![Number(3.), Number(1.), Binary(Minus), Number(2.), Binary(Times)]));
        assert_eq!(to_rpn(&[Number(1.), Binary(Minus), Unary(Minus), Unary(Minus), Number(2.)]),
                   Ok(vec![Number(1.), Number(2.), Unary(Minus), Unary(Minus), Binary(Minus)]));
        assert_eq!(to_rpn(&[Var("x".into()), Binary(Plus), Var("y".into())]),
                   Ok(vec![Var("x".into()), Var("y".into()), Binary(Plus)]));

        assert_eq!(to_rpn(&[Func("round".into()),
                            Func("sin".into()),
                            Number(1f64),
                            RParen,
                            RParen]),
                   Ok(vec![Number(1f64), Func("sin".into()), Func("round".into())]));

        assert_eq!(to_rpn(&[Binary(Plus)]), Err(RPNError::NotEnoughOperands(0)));
        assert_eq!(to_rpn(&[Var("x".into()), Number(1.)]),
                   Err(RPNError::TooManyOperands));
        assert_eq!(to_rpn(&[LParen]), Err(RPNError::MismatchedLParen(0)));
        assert_eq!(to_rpn(&[RParen]), Err(RPNError::MismatchedRParen(0)));
        assert_eq!(to_rpn(&[Func("sin".into())]),
                   Err(RPNError::MismatchedLParen(0)));
    }
}
