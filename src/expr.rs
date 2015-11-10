use std::ops::Deref;

use {Token, ParseError, RPNError};

/// Representain of an expression in the Reverse Polish notation form.
pub struct Expr {
    rpn: Vec<Token>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExprEvalError {
    UnknownVariable(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprError {
    ParseError(ParseError),
    RPNError(RPNError),
}

impl From<ParseError> for ExprError {
    fn from(err: ParseError) -> ExprError { ExprError::ParseError(err) }
}

impl From<RPNError> for ExprError {
    fn from(err: RPNError) -> ExprError { ExprError::RPNError(err) }
}

impl Expr {
    /// Constructs an expression by parsing a string.
    pub fn from_str<S: AsRef<str>>(string: S) -> Result<Expr, ExprError> {
        let tokens = try!(::tokenizer::tokenize(string));

        let rpn = try!(::shunting_yard::to_rpn(&tokens));

        Ok(Expr { rpn: rpn })
    }

    /// Evaluates the expression without any variables set.
    pub fn eval(&self) -> Result<f64, ExprEvalError> {
        use Token::*;
        use Operation::*;

        let mut stack = Vec::with_capacity(16);

        for token in &self.rpn {
            match *token {
                Var(ref n) => return Err(ExprEvalError::UnknownVariable(n.clone())),
                Number(f) => stack.push(f),
                Binary(op) => {
                    let right = stack.pop().unwrap();
                    let left = stack.pop().unwrap();
                    let r = match op {
                        Plus => left + right,
                        Minus => left - right,
                        Times => left * right,
                        Div => left / right,
                        Rem => left % right,
                        Pow => left.powf(right),
                    };
                    stack.push(r);
                }
                Unary(op) => {
                    let x = stack.pop().unwrap();
                    match op {
                        Plus => stack.push(x),
                        Minus => stack.push(-x),
                        _ => panic!("Unimplement unary operation: {:?}", op),
                    }
                }
                _ => panic!("Unrecognized token: {:?}", token),
            }
        }

        let r = stack.pop().expect("Stack is empty, this is impossible.");
        if !stack.is_empty() {
            panic!("There are still {} items on the stack.", stack.len());
        }
        Ok(r)
    }
}

impl Deref for Expr {
    type Target = [Token];

    fn deref(&self) -> &[Token] {
        &self.rpn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        assert_eq!(Expr::from_str("2 + 3").unwrap().eval(), Ok(5.));
        assert_eq!(Expr::from_str("2 + (3 + 4)").unwrap().eval(), Ok(9.));
        assert_eq!(Expr::from_str("-2^(4 - 3) * (3 + 4)").unwrap().eval(),
                   Ok(-14.));
        assert_eq!(Expr::from_str("a + 3").unwrap().eval(),
                   Err(ExprEvalError::UnknownVariable("a".into())));
    }
}
