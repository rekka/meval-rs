use std::ops::Deref;
use std::f64::consts;

use Error;
use tokenizer::{Token, tokenize};
use shunting_yard::to_rpn;
use std::fmt;

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
                Func(ref n, Some(i)) => {
                    if stack.len() < i {
                        panic!("eval: stack does not have enough arguments for function token {:?}", token);
                    }
                    match ctx.eval_func(n, &stack[stack.len() - i..]) {
                        Ok(r) => {
                            let nl = stack.len() - i;
                            stack.truncate(nl);
                        stack.push(r);}
                        Err(e) => return Err(Error::Function(n.to_owned(), e)),
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

    /// Creates a function of one variable based on this expression, with default constants and
    /// functions.
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
        try!(self.check_context(((var, 0.), &ctx)));
        let var = var.to_owned();
        return Ok(Box::new(move |x| self.eval(((&var, x), &ctx)).expect("Expr::bind")));
    }

    /// Creates a function of two variables based on this expression, with default constants and
    /// functions.
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
        try!(self.check_context(([(var1, 0.), (var2, 0.)], &ctx)));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        return Ok(Box::new(move |x, y| {
            self.eval(([(&var1, x), (&var2, y)], &ctx)).expect("Expr::bind2")
        }));
    }

    /// Creates a function of three variables based on this expression, with default constants and
    /// functions.
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
        try!(self.check_context(([(var1, 0.), (var2, 0.), (var3, 0.)], &ctx)));
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
    fn check_context<C: Context>(&self, ctx: C) -> Result<(), Error> {
        for t in self.rpn.iter() {
            match *t {
                Token::Var(ref name) => {
                if ctx.get_var(name).is_none() {
                    return Err(Error::UnknownVariable(name.clone()));
                }
                }
                Token::Func(ref name, Some(i)) => {
                    let v = vec![0.; i];
                    if let Err(e) = ctx.eval_func(name, &v) {
                        return Err(Error::Function(name.to_owned(), e));
                    }
                }
                Token::Func(_, None) => {
                    panic!("expr::check_context: Unexpected token: {:?}", *t);
                }
                Token::LParen | Token::RParen | Token::Binary(_) | Token::Unary(_) | Token::Comma | Token::Number(_) => {}
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
/// The built in context is given by the `builtin()` function (or the `Builtins` type).
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
///
/// // contexts can be combined using tuples
/// let ctx = (myvars, bins); // first context has preference if there's duplicity
///
/// assert_eq!(meval::eval_str_with_context("x * pi", ctx).unwrap(), 2. * std::f64::consts::PI);
/// ```
///
pub trait Context {
    fn get_var(&self, _: &str) -> Option<f64> {
        None
    }
    fn eval_func(&self, _: &str, _: &[f64]) -> Result<f64, FuncEvalError> {
        Err(FuncEvalError::UnknownFunction)
    }
}

/// Function evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum FuncEvalError {
    TooFewArguments,
    TooManyArguments,
    NumberArgs(usize),
    UnknownFunction,
}

impl fmt::Display for FuncEvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FuncEvalError::UnknownFunction =>
                write!(f, "Unknown function"),
            FuncEvalError::NumberArgs(i) =>
                write!(f, "Expected {} arguments", i),
            FuncEvalError::TooFewArguments => write!(f, "Too few arguments"),
            FuncEvalError::TooManyArguments => write!(f, "Too many arguments"),
        }
    }
}

/// Built-in functions and constants.
///
/// See the library documentation for the list of built-ins.
pub struct Builtins;

macro_rules! one_arg {
    ($args:expr, $func:ident) => {
        if $args.len() == 1 {
            Ok($args[0].$func())
        } else {
            Err(FuncEvalError::NumberArgs(1))
        }
    }
}

macro_rules! two_args {
    ($args:expr, $func:ident) => {
        if $args.len() == 2 {
            Ok($args[0].$func($args[1]))
        } else {
            Err(FuncEvalError::NumberArgs(2))
        }
    }
}

macro_rules! one_or_more_arg {
    ($args:expr, $func:ident) => {
        if $args.len() >= 1 {
            Ok($func($args))
        } else {
            Err(FuncEvalError::TooFewArguments)
        }
    }
}

fn max_array(xs: &[f64]) -> f64 {
    xs.iter().fold(::std::f64::NEG_INFINITY, |m, &x| m.max(x))
}

fn min_array(xs: &[f64]) -> f64 {
    xs.iter().fold(::std::f64::INFINITY, |m, &x| m.min(x))
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
            "atan2" => two_args!(args, atan2),
            "max" => one_or_more_arg!(args, max_array),
            "min" => one_or_more_arg!(args, min_array),
            _ => Err(FuncEvalError::UnknownFunction),
        }
    }
}

/// Returns the build-in constants in a form that can be used as a `Context`.
pub fn builtin() -> Builtins {
    // return [("pi", consts::PI), ("e", consts::E)];
    Builtins
}

impl<'a, T: Context> Context for &'a T {
    fn get_var(&self, name: &str) -> Option<f64> {
        (&**self).get_var(name)
    }

    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        (&**self).eval_func(name, args)
    }
}

impl<T: Context, S: Context> Context for (T, S) {
    fn get_var(&self, name: &str) -> Option<f64> {
        self.0.get_var(name).or_else(|| self.1.get_var(name))
    }
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        match self.0.eval_func(name, args) {
            Err(FuncEvalError::UnknownFunction) =>
                self.1.eval_func(name, args),
            e => e
        }
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

/// A custom function of one variable.
pub struct CustomFunc<S, T>(pub S, pub T);

/// A custom function of two variables.
pub struct CustomFunc2<S, T>(pub S, pub T);

/// A custom function of three variables.
pub struct CustomFunc3<S, T>(pub S, pub T);

/// A custom function of N variables.
pub struct CustomFuncN<S, T>(pub S, pub T, pub usize);

impl<S: AsRef<str>, T: Fn(f64) -> f64> Context for CustomFunc<S, T> {
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        if name != self.0.as_ref() {
            return Err(FuncEvalError::UnknownFunction);
        }
        if args.len() != 1 {
            return Err(FuncEvalError::NumberArgs(1));
        }
        Ok((self.1)(args[0]))
    }
}

impl<S: AsRef<str>, T: Fn(f64, f64) -> f64> Context for CustomFunc2<S, T> {
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        if name != self.0.as_ref() {
            return Err(FuncEvalError::UnknownFunction);
        }
        if args.len() != 2 {
            return Err(FuncEvalError::NumberArgs(2));
        }
        Ok((self.1)(args[0], args[1]))
    }
}

impl<S: AsRef<str>, T: Fn(f64, f64, f64) -> f64> Context for CustomFunc3<S, T> {
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        if name != self.0.as_ref() {
            return Err(FuncEvalError::UnknownFunction);
        }
        if args.len() != 3 {
            return Err(FuncEvalError::NumberArgs(3));
        }
        Ok((self.1)(args[0], args[1], args[2]))
    }
}

impl<S: AsRef<str>, T: Fn(&[f64]) -> f64> Context for CustomFuncN<S, T> {
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        if name != self.0.as_ref() {
            return Err(FuncEvalError::UnknownFunction);
        }
        if args.len() != self.2 {
            return Err(FuncEvalError::NumberArgs(self.2));
        }
        Ok((self.1)(args))
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
        assert_eq!(eval_str("max(1.)"), Ok(1.));
        assert_eq!(eval_str("max(1., 2., -1)"), Ok(2.));
        assert_eq!(eval_str("min(1., 2., -1)"), Ok(-1.));
        assert_eq!(eval_str("sin(1.) + cos(2.)"), Ok((1f64).sin() + (2f64).cos()));
    }

    #[test]
    fn test_builtins() {
        assert_eq!(eval_str("atan2(1.,2.)"), Ok((1f64).atan2(2.)));
    }

    #[test]
    fn test_eval_func_ctx() {
        let y = 5.;
        assert_eq!(eval_str_with_context("phi(2.)",
                                         CustomFunc("phi", |x| x + y + 3.)), Ok(2. + y + 3.));
        assert_eq!(eval_str_with_context("phi(2., 3.)",
                                         CustomFunc2("phi", |x, y| x + y + 3.)), Ok(2. + 3. + 3.));
        assert_eq!(eval_str_with_context("phi(2., 3., 4.)",
                                         CustomFunc3("phi", |x, y, z| x + y * z)), Ok(2. + 3. * 4.));
        assert_eq!(eval_str_with_context("phi(2., 3.)",
                                         CustomFuncN("phi", |xs: &[f64]| xs[0] + xs[1], 2)), Ok(2. + 3.));
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

        let expr = Expr::from_str("sin(x)").unwrap();
        let func = expr.clone().bind("x").unwrap();
        assert_eq!(func(1.), (1f64).sin());

        let expr = Expr::from_str("sin(x,2)").unwrap();
        match expr.clone().bind("x") {
            Err(Error::Function(_, FuncEvalError::NumberArgs(1))) => {},
            _ => panic!("bind did not error"),
        }
        let expr = Expr::from_str("hey(x,2)").unwrap();
        match expr.clone().bind("x") {
            Err(Error::Function(_, FuncEvalError::UnknownFunction)) => {},
            _ => panic!("bind did not error"),
        }
    }
}
