use Token;

#[derive(Debug, Clone, Copy)]
enum Associativity {
    Left,
    Right,
    NA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RPNError {
    MismatchedLParen(usize),
    MismatchedRParen(usize),
    NotEnoughOperands(usize),
    TooManyOperands,
}

/// Return operator precedence and associativity for a given token.
fn prec_assoc(token: &Token) -> (u32, Associativity) {
    use self::Associativity::*;
    use Token::*;
    use Operation::*;
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
        Var(_) | Number(_) | LParen | RParen => (0, NA),
    }
}


/// Convert a tokenized infix expression to a Reverse Polish notation.
pub fn to_rpn(input: &[Token]) -> Result<Vec<Token>, RPNError> {
    use Token::*;

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
                        _ => output.push(t),
                    }
                }
                if !found {
                    return Err(RPNError::MismatchedRParen(index));
                }
            }
        }
    }

    while let Some((index, token)) = stack.pop() {
        match token {
            Unary(_) | Binary(_) => output.push(token),
            LParen => return Err(RPNError::MismatchedLParen(index)),
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
    use Token::*;
    use Operation::*;

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

        assert_eq!(to_rpn(&[Binary(Plus)]), Err(RPNError::NotEnoughOperands(0)));
        assert_eq!(to_rpn(&[Var("x".into()), Number(1.)]),
                   Err(RPNError::TooManyOperands));
        assert_eq!(to_rpn(&[LParen]), Err(RPNError::MismatchedLParen(0)));
        assert_eq!(to_rpn(&[RParen]), Err(RPNError::MismatchedRParen(0)));
    }
}
