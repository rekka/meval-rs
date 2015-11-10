use std::ops::{Deref, DerefMut};
use std::collections::BTreeMap;
use std::f64::consts;

use Error;
use tokenizer::{Token, tokenize};
use shunting_yard::to_rpn;

/// Representain of an expression in the Reverse Polish notation form.
pub struct Expr {
    rpn: Vec<Token>,
}

impl Expr {
    /// Constructs an expression by parsing a string.
    pub fn from_str<S: AsRef<str>>(string: S) -> Result<Expr, Error> {
        let tokens = try!(tokenize(string));

        let rpn = try!(to_rpn(&tokens));

        Ok(Expr { rpn: rpn })
    }

    /// Evaluates the expression with variables given by the argument.
    pub fn eval<C: ExprContextProvider>(&self, ctx: C) -> Result<f64, Error> {
        use tokenizer::Token::*;
        use tokenizer::Operation::*;

        let mut stack = Vec::with_capacity(16);

        for token in &self.rpn {
            match *token {
                Var(ref n) =>
                    if let Some(v) = ctx.get_var(n) {
                        stack.push(v);
                    } else {
                        return Err(Error::UnknownVariable(n.clone()));
                    },
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

/// Evaluate a string with default constants.
pub fn eval_str<S: AsRef<str>>(expr: S) -> Result<f64, Error> {
    let expr = try!(Expr::from_str(expr));

    expr.eval(ExprContext::new())
}

/// Evaluate a string with default constants.
pub fn eval_str_with_context<S: AsRef<str>, C: ExprContextProvider>(expr: S, ctx: C) -> Result<f64, Error> {
    let expr = try!(Expr::from_str(expr));

    expr.eval(ctx)
}

impl Deref for Expr {
    type Target = [Token];

    fn deref(&self) -> &[Token] {
        &self.rpn
    }
}

pub trait ExprContextProvider {
    fn get_var(&self, name: &str) -> Option<f64>;
}

#[derive(Debug, Clone)]
pub struct ExprContext {
    vars: BTreeMap<String, f64>,
}

impl ExprContext {
    pub fn new() -> ExprContext {
        let mut vars = BTreeMap::new();
        vars.insert("pi".into(), consts::PI);
        vars.insert("e".into(), consts::E);

        ExprContext { vars: vars }
    }

    pub fn without_default() -> ExprContext {
        ExprContext { vars: BTreeMap::new() }
    }
}

impl Deref for ExprContext {
    type Target = BTreeMap<String, f64>;
    fn deref(&self) -> &BTreeMap<String, f64> {
        &self.vars
    }
}

impl DerefMut for ExprContext {
    fn deref_mut(&mut self) -> &mut BTreeMap<String, f64> {
        &mut self.vars
    }
}

impl Default for ExprContext {
    fn default() -> ExprContext {
        ExprContext::new()
    }
}

impl ExprContextProvider for ExprContext {
    fn get_var(&self, name: &str) -> Option<f64> {
        self.get(name).map(|f| *f)
    }
}

impl<'a, T: ExprContextProvider> ExprContextProvider for &'a T {
    fn get_var(&self, name: &str) -> Option<f64> {
        (&**self).get_var(name)
    }
}

macro_rules! arg {
    () => {
        $crate::ExprContext::new()
    };

    ($var:ident: $value:expr) => {
        {
            let mut ctx = $crate::ExprContext::new();
            ctx.insert(stringify!($var).into(), $value);
            ctx
        }
    };

    ($($var:ident: $value:expr),*) => {
        {
            let mut ctx = $crate::ExprContext::new();
            $(
                ctx.insert(stringify!($var).into(), $value);
            )*
            ctx
        }
    };
    ($($var:ident: $value:expr),*,) => {
        arg!($($var: $value),*)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use Error;

    #[test]
    fn test_eval() {
        assert_eq!(eval_str("2 + 3"), Ok(5.));
        assert_eq!(eval_str("2 + (3 + 4)"), Ok(9.));
        assert_eq!(eval_str("-2^(4 - 3) * (3 + 4)"), Ok(-14.));
        assert_eq!(eval_str("a + 3"), Err(Error::UnknownVariable("a".into())));
        assert_eq!(eval_str_with_context("a + 3", arg! {a: 2.}), Ok(5.));
        assert_eq!(eval_str_with_context("hey ^ no", arg! {hey: 2., no: 8.}), Ok(256.));
    }
}
