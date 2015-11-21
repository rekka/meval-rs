use std::ops::Deref;
use std::f64::consts;

use Error;
use tokenizer::{Token, tokenize};
use shunting_yard::to_rpn;

/// Representation of a parsed expression.
///
/// The expression is internally stored in [reverse Polish notation (RPN)][RPN] as a sequence of
/// `Token`s.
///
/// Methods `bind`, `bind_with_context`, `bind2`, ... can be used to create (boxed) closures from
/// the expression that then can be passed around and used as any other `Fn` closures.  A boxed
/// closure is unfortunately currently slightly less convenient than an unboxed closure since
/// `Box<Fn(f64) -> f64>` does not implement `FnOnce`, `Fn` or `FnMut`. So to use it directly as a
/// function argument where a closure is expected, it has to be manually dereferenced:
///
/// ```rust
/// let func = meval::Expr::from_str("x").unwrap().bind("x").unwrap();
/// let r = Some(2.).map(&*func);
/// ```
///
/// [RPN]: https://en.wikipedia.org/wiki/Reverse_Polish_notation
#[derive(Debug, Clone)]
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
    pub fn eval<C: Context>(&self, ctx: C) -> Result<f64, Error> {
        use tokenizer::Token::*;
        use tokenizer::Operation::*;

        let mut stack = Vec::with_capacity(16);

        for token in &self.rpn {
            match *token {
                Var(ref n) => if let Some(v) = ctx.get_var(n) {
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
                Func(ref n, Some(1)) => {
                    let arg = stack.pop().unwrap();
                    let r = match n.as_ref() {
                        "sqrt" => arg.sqrt(),
                        "exp" => arg.exp(),
                        "ln" => arg.ln(),
                        "abs" => arg.abs(),
                        "sin" => arg.sin(),
                        "cos" => arg.cos(),
                        "tan" => arg.tan(),
                        "asin" => arg.asin(),
                        "acos" => arg.acos(),
                        "atan" => arg.atan(),
                        "sinh" => arg.sinh(),
                        "cosh" => arg.cosh(),
                        "tanh" => arg.tanh(),
                        "asinh" => arg.asinh(),
                        "acosh" => arg.acosh(),
                        "atanh" => arg.atanh(),
                        "floor" => arg.floor(),
                        "ceil" => arg.ceil(),
                        "round" => arg.round(),
                        "signum" => arg.signum(),
                        _ => return Err(Error::UnknownFunction(n.clone())),
                    };
                    stack.push(r);
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

    /// Creates a function of one variable based on this expression, with default constants.
    ///
    /// Binds the input of the returned closure to `var`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by the default
    /// context or `var`.
    pub fn bind<'a>(self, var: &str) -> Result<Box<Fn(f64) -> f64 + 'a>, Error> {
        return self.bind_with_context(builtin(), var);
    }

    /// Creates a function of one variable based on this expression.
    ///
    /// Binds the input of the returned closure to `var`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind_with_context<'a, C>(self,
                                    ctx: C,
                                    var: &str)
                                    -> Result<Box<Fn(f64) -> f64 + 'a>, Error>
        where C: Context + 'a
    {
        try!(self.check_vars(((var, 0.), &ctx)));
        let var = var.to_owned();
        return Ok(Box::new(move |x| self.eval(((&var, x), &ctx)).expect("Expr::bind")));
    }

    /// Creates a function of two variables based on this expression, with default constants.
    ///
    /// Binds the inputs of the returned closure to `var1` and `var2`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by the default
    /// context or `var`.
    pub fn bind2<'a>(self, var1: &str, var2: &str) -> Result<Box<Fn(f64, f64) -> f64 + 'a>, Error> {
        return self.bind2_with_context(builtin(), var1, var2);
    }

    /// Creates a function of two variables based on this expression.
    ///
    /// Binds the inputs of the returned closure to `var1` and `var2`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind2_with_context<'a, C>(self,
                                     ctx: C,
                                     var1: &str,
                                     var2: &str)
                                     -> Result<Box<Fn(f64, f64) -> f64 + 'a>, Error>
        where C: Context + 'a
    {
        try!(self.check_vars(([(var1, 0.), (var2, 0.)], &ctx)));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        return Ok(Box::new(move |x, y| {
            self.eval(([(&var1, x), (&var2, y)], &ctx)).expect("Expr::bind2")
        }));
    }

    /// Creates a function of three variables based on this expression, with default constants.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2` and `var3`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by the default
    /// context or `var`.
    pub fn bind3<'a>(self,
                     var1: &str,
                     var2: &str,
                     var3: &str)
                     -> Result<Box<Fn(f64, f64, f64) -> f64 + 'a>, Error> {
        return self.bind3_with_context(builtin(), var1, var2, var3);
    }

    /// Creates a function of three variables based on this expression.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2` and `var3`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind3_with_context<'a, C>(self,
                                     ctx: C,
                                     var1: &str,
                                     var2: &str,
                                     var3: &str)
                                     -> Result<Box<Fn(f64, f64, f64) -> f64 + 'a>, Error>
        where C: Context + 'a
    {
        try!(self.check_vars(([(var1, 0.), (var2, 0.), (var3, 0.)], &ctx)));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        let var3 = var3.to_owned();
        return Ok(Box::new(move |x, y, z| {
            self.eval(([(&var1, x), (&var2, y), (&var3, z)], &ctx)).expect("Expr::bind3")
        }));
    }

    /// Checks that the value of every variable in the expression is specified by the context `ctx`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if a missing variable is detected.
    fn check_vars<C: Context>(&self, ctx: C) -> Result<(), Error> {
        for t in self.rpn.iter() {
            if let &Token::Var(ref name) = t {
                if ctx.get_var(name).is_none() {
                    return Err(Error::UnknownVariable(name.clone()));
                }
            }
        }
        Ok(())
    }
}

/// Evaluates a string with built-in constants.
pub fn eval_str<S: AsRef<str>>(expr: S) -> Result<f64, Error> {
    let expr = try!(Expr::from_str(expr));

    expr.eval(builtin())
}

/// Evaluates a string with the given context.
///
/// No build-ins are defined in this case.
pub fn eval_str_with_context<S: AsRef<str>, C: Context>(expr: S, ctx: C) -> Result<f64, Error> {
    let expr = try!(Expr::from_str(expr));

    expr.eval(ctx)
}

impl Deref for Expr {
    type Target = [Token];

    fn deref(&self) -> &[Token] {
        &self.rpn
    }
}

/// Values of variables (and constants) for substitution into an evaluated expression.
///
/// A `Context` can be built from other contexts:
///
/// ```rust
/// use meval::Context;
///
/// let bins = meval::builtin(); // built-ins
/// assert_eq!(bins.get_var("pi"), Some(std::f64::consts::PI));
///
/// let myvars = ("x", 2.);
/// assert_eq!(myvars.get_var("x"), Some(2f64));
/// let ctx = (myvars, bins); // first context has preference if there's duplicity
///
/// assert_eq!(meval::eval_str_with_context("x * pi", ctx).unwrap(), 2. * std::f64::consts::PI);
/// ```
///
pub trait Context {
    fn get_var(&self, name: &str) -> Option<f64> {
        None
    }
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        Err(FuncEvalError::UnknownFunction)
    }
}

enum FuncEvalError {
    TooFewArguments,
    TooManyArguments,
    NumberArgs(usize),
    UnknownFunction,
}

struct Builtins;

macro_rules! one_arg {
    ($args:expr, $func:ident) => {
        if $args.len() == 1 {
            Ok($args[0].$func())
        } else {
            Err(FuncEvalError::NumberArgs(1))
        }
    }
}

impl Context for Builtins {
    fn get_var(&self, name: &str) -> Option<f64> {
        match name {
            "pi" => Some(consts::PI),
            "e" => Some(consts::E),
            _ => None,
        }
    }
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        match name {
            "sqrt" => one_arg!(args, sqrt),
            "exp" => one_arg!(args, exp),
            "ln" => one_arg!(args, ln),
            "abs" => one_arg!(args, abs),
            "sin" => one_arg!(args, sin),
            "cos" => one_arg!(args, cos),
            "tan" => one_arg!(args, tan),
            "asin" => one_arg!(args, asin),
            "acos" => one_arg!(args, acos),
            "atan" => one_arg!(args, atan),
            "sinh" => one_arg!(args, sinh),
            "cosh" => one_arg!(args, cosh),
            "tanh" => one_arg!(args, tanh),
            "asinh" => one_arg!(args, asinh),
            "acosh" => one_arg!(args, acosh),
            "atanh" => one_arg!(args, atanh),
            "floor" => one_arg!(args, floor),
            "ceil" => one_arg!(args, ceil),
            "round" => one_arg!(args, round),
            "signum" => one_arg!(args, signum),
            _ => Err(FuncEvalError::UnknownFunction),
        }
    }
}

/// Returns the build-in constants in a form that can be used as a `Context`.
pub fn builtin() -> [(&'static str, f64); 2] {
    return [("pi", consts::PI), ("e", consts::E)];
}

impl<'a, T: Context> Context for &'a T {
    fn get_var(&self, name: &str) -> Option<f64> {
        (&**self).get_var(name)
    }
}

impl<T: Context, S: Context> Context for (T, S) {
    fn get_var(&self, name: &str) -> Option<f64> {
        self.0.get_var(name).or_else(|| self.1.get_var(name))
    }
}

impl<S: AsRef<str>> Context for (S, f64) {
    fn get_var(&self, name: &str) -> Option<f64> {
        if self.0.as_ref() == name {
            Some(self.1)
        } else {
            None
        }
    }
}

// macro for implementing Context for arrays
macro_rules! array_impls {
    ($($N:expr)+) => {
        $(
            impl<S: AsRef<str>> Context for [(S, f64); $N] {
                fn get_var(&self, name: &str) -> Option<f64> {
                    for &(ref n, v) in self.iter() {
                        if n.as_ref() == name {
                            return Some(v);
                        }
                    }
                    None
                }
            }
        )+
    }
}

array_impls! {
    0 1 2 3 4 5 6 7 8
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
        // assert_eq!(eval_str_with_context("a + 3", arg! {a: 2.}), Ok(5.));
        // assert_eq!(eval_str_with_context("hey ^ no", arg! {hey: 2., no: 8.}),
        //            Ok(256.));
        assert_eq!(eval_str("round(sin (pi) * cos(0))"), Ok(0.));
        assert_eq!(eval_str("round( sqrt(3^2 + 4^2)) "), Ok(5.));
    }

    #[test]
    fn test_bind() {
        let expr = Expr::from_str("x + 3").unwrap();
        let func = expr.clone().bind("x").unwrap();
        assert_eq!(func(1.), 4.);

        assert_eq!(expr.clone().bind("y").err(),
                   Some(Error::UnknownVariable("x".into())));

        let ctx = (("x", 2.), builtin());
        let func = expr.bind_with_context(&ctx, "y").unwrap();
        assert_eq!(func(1.), 5.);

        let expr = Expr::from_str("x + y + 2.").unwrap();
        let func = expr.clone().bind2("x", "y").unwrap();
        assert_eq!(func(1., 2.), 5.);
        assert_eq!(expr.clone().bind2("z", "y").err(),
                   Some(Error::UnknownVariable("x".into())));
        assert_eq!(expr.bind2("x", "z").err(),
                   Some(Error::UnknownVariable("y".into())));

        let expr = Expr::from_str("x + y^2 + z^3").unwrap();
        let func = expr.clone().bind3("x", "y", "z").unwrap();
        assert_eq!(func(1., 2., 3.), 32.);
    }
}
