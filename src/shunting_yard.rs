use Token;

#[derive(Clone, Copy)]
enum Associativity {
    Left,
    Right,
    NA,
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
pub fn to_rpn(input: &[Token]) -> Vec<Token> {
    use Token::*;

    let mut output = Vec::with_capacity(input.len());
    let mut stack = Vec::with_capacity(input.len());

    for token in input {
        let token = token.clone();
        match token {
            Number(_) | Var(_) => output.push(token),
            Unary(_) => stack.push(token),
            Binary(_) => {
                let pa1 = prec_assoc(&token);
                while !stack.is_empty() {
                    let pa2 = prec_assoc(stack.last().unwrap());
                    match (pa1, pa2) {
                        ((i, Associativity::Left), (j, _)) if i <= j => {
                            output.push(stack.pop().unwrap());
                        }
                        ((i, Associativity::Right), (j, _)) if i < j => {
                            output.push(stack.pop().unwrap());
                        }
                        _ => {
                            break;
                        }
                    }
                }
                stack.push(token)
            }
            LParen => stack.push(token),
            RParen => {
                let mut found = false;
                while let Some(t) = stack.pop() {
                    match t {
                        LParen => {
                            found = true;
                            break;
                        }
                        _ => output.push(t),
                    }
                }
                if !found {
                    panic!("Mismatched right parenthesis.");
                }
            }
        }
    }

    while let Some(token) = stack.pop() {
        match token {
            Unary(_) | Binary(_) => output.push(token),
            LParen => panic!("Mismatched left parenthesis."),
            _ => panic!("Unexpected token on stack."),
        }
    }

    output.shrink_to_fit();
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use Token::*;
    use Operation::*;

    #[test]
    fn test_to_rpn() {
        assert_eq!(to_rpn(&[Number(1.)]), vec![Number(1.)]);
        assert_eq!(to_rpn(&[Number(1.), Binary(Plus), Number(2.)]),
                   vec![Number(1.), Number(2.), Binary(Plus)]);
        assert_eq!(to_rpn(&[Unary(Minus), Number(1.), Binary(Pow), Number(2.)]),
                   vec![Number(1.), Number(2.), Binary(Pow), Unary(Minus)]);
        assert_eq!(to_rpn(&[Number(3.), Binary(Minus), Number(1.), Binary(Times), Number(2.)]),
                   vec![Number(3.), Number(1.), Number(2.), Binary(Times), Binary(Minus)]);
        assert_eq!(to_rpn(&[LParen,
                            Number(3.),
                            Binary(Minus),
                            Number(1.),
                            RParen,
                            Binary(Times),
                            Number(2.)]),
                   vec![Number(3.), Number(1.), Binary(Minus), Number(2.), Binary(Times)]);
        assert_eq!(to_rpn(&[Number(1.), Binary(Minus), Unary(Minus), Unary(Minus), Number(2.)]),
                   vec![Number(1.), Number(2.), Unary(Minus), Unary(Minus), Binary(Minus)]);
        assert_eq!(to_rpn(&[Var("x".into()), Binary(Plus), Var("y".into())]), vec![Var("x".into()), Var("y".into()), Binary(Plus)]);
    }
}
