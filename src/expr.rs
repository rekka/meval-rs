use fnv::FnvHashMap;
use std::f64::consts;
use std::ops::Deref;
use std::rc::Rc;
use std::str::FromStr;

type ContextHashMap<K, V> = FnvHashMap<K, V>;

use shunting_yard::to_rpn;
use std;
use std::fmt;
use tokenizer::{tokenize, Token};
use Error;

/// Representation of a parsed expression.
///
/// The expression is internally stored in the [reverse Polish notation (RPN)][RPN] as a sequence
/// of `Token`s.
///
/// Methods `bind`, `bind_with_context`, `bind2`, ... can be used to create  closures from
/// the expression that then can be passed around and used as any other `Fn` closures.
///
/// ```rust
/// let func = "x^2".parse::<meval::Expr>().unwrap().bind("x").unwrap();
/// let r = Some(2.).map(func);
/// assert_eq!(r, Some(4.));
/// ```
///
/// [RPN]: https://en.wikipedia.org/wiki/Reverse_Polish_notation
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    rpn: Vec<Token>,
}

impl Expr {
    /// Evaluates the expression.
    pub fn eval(&self) -> Result<f64, Error> {
        self.eval_with_context(builtin())
    }

    /// Evaluates the expression with variables given by the argument.
    pub fn eval_with_context<C: ContextProvider>(&self, ctx: C) -> Result<f64, Error> {
        use tokenizer::Operation::*;
        use tokenizer::Token::*;

        let mut stack = Vec::with_capacity(16);

        for token in &self.rpn {
            match *token {
                Var(ref n) => {
                    if let Some(v) = ctx.get_var(n) {
                        stack.push(v);
                    } else {
                        return Err(Error::UnknownVariable(n.clone()));
                    }
                }
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
                        panic!(
                            "eval: stack does not have enough arguments for function token \
                             {:?}",
                            token
                        );
                    }
                    match ctx.eval_func(n, &stack[stack.len() - i..]) {
                        Ok(r) => {
                            let nl = stack.len() - i;
                            stack.truncate(nl);
                            stack.push(r);
                        }
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
    pub fn bind<'a>(self, var: &str) -> Result<impl Fn(f64) -> f64 + 'a, Error> {
        self.bind_with_context(builtin(), var)
    }

    /// Creates a function of one variable based on this expression.
    ///
    /// Binds the input of the returned closure to `var`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind_with_context<'a, C>(
        self,
        ctx: C,
        var: &str,
    ) -> Result<impl Fn(f64) -> f64 + 'a, Error>
    where
        C: ContextProvider + 'a,
    {
        try!(self.check_context(((var, 0.), &ctx)));
        let var = var.to_owned();
        Ok(move |x| {
            self.eval_with_context(((&var, x), &ctx))
                .expect("Expr::bind")
        })
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
    pub fn bind2<'a>(self, var1: &str, var2: &str) -> Result<impl Fn(f64, f64) -> f64 + 'a, Error> {
        self.bind2_with_context(builtin(), var1, var2)
    }

    /// Creates a function of two variables based on this expression.
    ///
    /// Binds the inputs of the returned closure to `var1` and `var2`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind2_with_context<'a, C>(
        self,
        ctx: C,
        var1: &str,
        var2: &str,
    ) -> Result<impl Fn(f64, f64) -> f64 + 'a, Error>
    where
        C: ContextProvider + 'a,
    {
        try!(self.check_context(([(var1, 0.), (var2, 0.)], &ctx)));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        Ok(move |x, y| {
            self.eval_with_context(([(&var1, x), (&var2, y)], &ctx))
                .expect("Expr::bind2")
        })
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
    pub fn bind3<'a>(
        self,
        var1: &str,
        var2: &str,
        var3: &str,
    ) -> Result<impl Fn(f64, f64, f64) -> f64 + 'a, Error> {
        self.bind3_with_context(builtin(), var1, var2, var3)
    }

    /// Creates a function of three variables based on this expression.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2` and `var3`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind3_with_context<'a, C>(
        self,
        ctx: C,
        var1: &str,
        var2: &str,
        var3: &str,
    ) -> Result<impl Fn(f64, f64, f64) -> f64 + 'a, Error>
    where
        C: ContextProvider + 'a,
    {
        try!(self.check_context(([(var1, 0.), (var2, 0.), (var3, 0.)], &ctx)));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        let var3 = var3.to_owned();
        Ok(move |x, y, z| {
            self.eval_with_context(([(&var1, x), (&var2, y), (&var3, z)], &ctx))
                .expect("Expr::bind3")
        })
    }

    /// Creates a function of four variables based on this expression, with default constants and
    /// functions.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2`, `var3` and `var4`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by the default
    /// context or `var`.
    pub fn bind4<'a>(
        self,
        var1: &str,
        var2: &str,
        var3: &str,
        var4: &str,
    ) -> Result<impl Fn(f64, f64, f64, f64) -> f64 + 'a, Error> {
        self.bind4_with_context(builtin(), var1, var2, var3, var4)
    }

    /// Creates a function of four variables based on this expression.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2`, `var3` and `var4`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind4_with_context<'a, C>(
        self,
        ctx: C,
        var1: &str,
        var2: &str,
        var3: &str,
        var4: &str,
    ) -> Result<impl Fn(f64, f64, f64, f64) -> f64 + 'a, Error>
    where
        C: ContextProvider + 'a,
    {
        try!(self.check_context(([(var1, 0.), (var2, 0.), (var3, 0.), (var4, 0.)], &ctx)));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        let var3 = var3.to_owned();
        let var4 = var4.to_owned();
        Ok(move |x1, x2, x3, x4| {
            self.eval_with_context(([(&var1, x1), (&var2, x2), (&var3, x3), (&var4, x4)], &ctx))
                .expect("Expr::bind4")
        })
    }

    /// Creates a function of five variables based on this expression, with default constants and
    /// functions.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2`, `var3`, `var4` and `var5`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by the default
    /// context or `var`.
    pub fn bind5<'a>(
        self,
        var1: &str,
        var2: &str,
        var3: &str,
        var4: &str,
        var5: &str,
    ) -> Result<impl Fn(f64, f64, f64, f64, f64) -> f64 + 'a, Error> {
        self.bind5_with_context(builtin(), var1, var2, var3, var4, var5)
    }

    /// Creates a function of five variables based on this expression.
    ///
    /// Binds the inputs of the returned closure to `var1`, `var2`, `var3`, `var4` and `var5`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bind5_with_context<'a, C>(
        self,
        ctx: C,
        var1: &str,
        var2: &str,
        var3: &str,
        var4: &str,
        var5: &str,
    ) -> Result<impl Fn(f64, f64, f64, f64, f64) -> f64 + 'a, Error>
    where
        C: ContextProvider + 'a,
    {
        try!(self.check_context((
            [(var1, 0.), (var2, 0.), (var3, 0.), (var4, 0.), (var5, 0.)],
            &ctx
        )));
        let var1 = var1.to_owned();
        let var2 = var2.to_owned();
        let var3 = var3.to_owned();
        let var4 = var4.to_owned();
        let var5 = var5.to_owned();
        Ok(move |x1, x2, x3, x4, x5| {
            self.eval_with_context((
                [
                    (&var1, x1),
                    (&var2, x2),
                    (&var3, x3),
                    (&var4, x4),
                    (&var5, x5),
                ],
                &ctx,
            )).expect("Expr::bind5")
        })
    }

    /// Binds the input of the returned closure to elements of `vars`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by the default
    /// context or `var`.
    pub fn bindn<'a>(self, vars: &'a [&str]) -> Result<impl Fn(&[f64]) -> f64 + 'a, Error> {
        self.bindn_with_context(builtin(), vars)
    }

    /// Creates a function of N variables based on this expression.
    ///
    /// Binds the input of the returned closure to the elements of `vars`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if there is a variable in the expression that is not provided by `ctx` or
    /// `var`.
    pub fn bindn_with_context<'a, C>(
        self,
        ctx: C,
        vars: &'a [&str],
    ) -> Result<impl Fn(&[f64]) -> f64 + 'a, Error>
    where
        C: ContextProvider + 'a,
    {
        let n = vars.len();
        try!(
            self.check_context((
                vars.into_iter()
                    .zip(vec![0.; n].into_iter())
                    .collect::<Vec<_>>(),
                &ctx
            ))
        );
        let vars = vars.iter().map(|v| v.to_owned()).collect::<Vec<_>>();
        Ok(move |x: &[f64]| {
            self.eval_with_context((
                vars.iter()
                    .zip(x.into_iter())
                    .map(|(v, x)| (v, *x))
                    .collect::<Vec<_>>(),
                &ctx,
            )).expect("Expr::bindn")
        })
    }

    /// Checks that the value of every variable in the expression is specified by the context `ctx`.
    ///
    /// # Failure
    ///
    /// Returns `Err` if a missing variable is detected.
    fn check_context<C: ContextProvider>(&self, ctx: C) -> Result<(), Error> {
        for t in &self.rpn {
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
                Token::LParen
                | Token::RParen
                | Token::Binary(_)
                | Token::Unary(_)
                | Token::Comma
                | Token::Number(_) => {}
            }
        }
        Ok(())
    }
}

/// Evaluates a string with built-in constants and functions.
pub fn eval_str<S: AsRef<str>>(expr: S) -> Result<f64, Error> {
    let expr = try!(Expr::from_str(expr.as_ref()));

    expr.eval_with_context(builtin())
}

impl FromStr for Expr {
    type Err = Error;
    /// Constructs an expression by parsing a string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = try!(tokenize(s));

        let rpn = try!(to_rpn(&tokens));

        Ok(Expr { rpn: rpn })
    }
}

/// Evaluates a string with the given context.
///
/// No built-ins are defined in this case.
pub fn eval_str_with_context<S: AsRef<str>, C: ContextProvider>(
    expr: S,
    ctx: C,
) -> Result<f64, Error> {
    let expr = try!(Expr::from_str(expr.as_ref()));

    expr.eval_with_context(ctx)
}

impl Deref for Expr {
    type Target = [Token];

    fn deref(&self) -> &[Token] {
        &self.rpn
    }
}

/// A trait of a source of variables (and constants) and functions for substitution into an
/// evaluated expression.
///
/// A simplest way to create a custom context provider is to use [`Context`](struct.Context.html).
///
/// ## Advanced usage
///
/// Alternatively, values of variables/constants can be specified by tuples `(name, value)`,
/// `std::collections::HashMap` or `std::collections::BTreeMap`.
///
/// ```rust
/// use meval::{ContextProvider, Context};
///
/// let mut ctx = Context::new(); // built-ins
/// ctx.var("x", 2.); // insert a new variable
/// assert_eq!(ctx.get_var("pi"), Some(std::f64::consts::PI));
///
/// let myvars = ("x", 2.); // tuple as a ContextProvider
/// assert_eq!(myvars.get_var("x"), Some(2f64));
///
/// // HashMap as a ContextProvider
/// let mut varmap = std::collections::HashMap::new();
/// varmap.insert("x", 2.);
/// varmap.insert("y", 3.);
/// assert_eq!(varmap.get_var("x"), Some(2f64));
/// assert_eq!(varmap.get_var("z"), None);
/// ```
///
/// Custom functions can be also defined.
///
/// ```rust
/// use meval::{ContextProvider, Context};
///
/// let mut ctx = Context::new(); // built-ins
/// ctx.func2("phi", |x, y| x / (y * y));
///
/// assert_eq!(ctx.eval_func("phi", &[2., 3.]), Ok(2. / (3. * 3.)));
/// ```
///
/// A `ContextProvider` can be built by combining other contexts:
///
/// ```rust
/// use meval::Context;
///
/// let bins = Context::new(); // built-ins
/// let mut funcs = Context::empty(); // empty context
/// funcs.func2("phi", |x, y| x / (y * y));
/// let myvars = ("x", 2.);
///
/// // contexts can be combined using tuples
/// let ctx = ((myvars, bins), funcs); // first context has preference if there's duplicity
///
/// assert_eq!(meval::eval_str_with_context("x * pi + phi(1., 2.)", ctx).unwrap(), 2. *
///             std::f64::consts::PI + 1. / (2. * 2.));
/// ```
///
pub trait ContextProvider {
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
            FuncEvalError::UnknownFunction => write!(f, "Unknown function"),
            FuncEvalError::NumberArgs(i) => write!(f, "Expected {} arguments", i),
            FuncEvalError::TooFewArguments => write!(f, "Too few arguments"),
            FuncEvalError::TooManyArguments => write!(f, "Too many arguments"),
        }
    }
}

impl std::error::Error for FuncEvalError {
    fn description(&self) -> &str {
        match *self {
            FuncEvalError::UnknownFunction => "unknown function",
            FuncEvalError::NumberArgs(_) => "wrong number of function arguments",
            FuncEvalError::TooFewArguments => "too few function arguments",
            FuncEvalError::TooManyArguments => "too many function arguments",
        }
    }
}

#[doc(hidden)]
pub fn max_array(xs: &[f64]) -> f64 {
    xs.iter().fold(::std::f64::NEG_INFINITY, |m, &x| m.max(x))
}

#[doc(hidden)]
pub fn min_array(xs: &[f64]) -> f64 {
    xs.iter().fold(::std::f64::INFINITY, |m, &x| m.min(x))
}

/// Returns the built-in constants and functions in a form that can be used as a `ContextProvider`.
#[doc(hidden)]
pub fn builtin<'a>() -> Context<'a> {
    // TODO: cache this (lazy_static)
    Context::new()
}

impl<'a, T: ContextProvider> ContextProvider for &'a T {
    fn get_var(&self, name: &str) -> Option<f64> {
        (&**self).get_var(name)
    }

    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        (&**self).eval_func(name, args)
    }
}

impl<'a, T: ContextProvider> ContextProvider for &'a mut T {
    fn get_var(&self, name: &str) -> Option<f64> {
        (&**self).get_var(name)
    }

    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        (&**self).eval_func(name, args)
    }
}

impl<T: ContextProvider, S: ContextProvider> ContextProvider for (T, S) {
    fn get_var(&self, name: &str) -> Option<f64> {
        self.0.get_var(name).or_else(|| self.1.get_var(name))
    }
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        match self.0.eval_func(name, args) {
            Err(FuncEvalError::UnknownFunction) => self.1.eval_func(name, args),
            e => e,
        }
    }
}

impl<S: AsRef<str>> ContextProvider for (S, f64) {
    fn get_var(&self, name: &str) -> Option<f64> {
        if self.0.as_ref() == name {
            Some(self.1)
        } else {
            None
        }
    }
}

/// `std::collections::HashMap` of variables.
impl<S> ContextProvider for std::collections::HashMap<S, f64>
where
    S: std::hash::Hash + std::cmp::Eq + std::borrow::Borrow<str>,
{
    fn get_var(&self, name: &str) -> Option<f64> {
        self.get(name).cloned()
    }
}

/// `std::collections::BTreeMap` of variables.
impl<S> ContextProvider for std::collections::BTreeMap<S, f64>
where
    S: std::cmp::Ord + std::borrow::Borrow<str>,
{
    fn get_var(&self, name: &str) -> Option<f64> {
        self.get(name).cloned()
    }
}

impl<S: AsRef<str>> ContextProvider for Vec<(S, f64)> {
    fn get_var(&self, name: &str) -> Option<f64> {
        for &(ref n, v) in self.iter() {
            if n.as_ref() == name {
                return Some(v);
            }
        }
        None
    }
}

// macro for implementing ContextProvider for arrays
macro_rules! array_impls {
    ($($N:expr)+) => {
        $(
            impl<S: AsRef<str>> ContextProvider for [(S, f64); $N] {
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

/// A structure for storing variables/constants and functions to be used in an expression.
///
/// # Example
///
/// ```rust
/// use meval::{eval_str_with_context, Context};
///
/// let mut ctx = Context::new(); // builtins
/// ctx.var("x", 3.)
///    .func("f", |x| 2. * x)
///    .funcn("sum", |xs| xs.iter().sum(), ..);
///
/// assert_eq!(eval_str_with_context("pi + sum(1., 2.) + f(x)", &ctx),
///            Ok(std::f64::consts::PI + 1. + 2. + 2. * 3.));
/// ```
#[derive(Clone)]
pub struct Context<'a> {
    vars: ContextHashMap<String, f64>,
    funcs: ContextHashMap<String, GuardedFunc<'a>>,
}

impl<'a> Context<'a> {
    /// Creates a context with built-in constants and functions.
    pub fn new() -> Context<'a> {
        thread_local!(static DEFAULT_CONTEXT: Context<'static> = {
            let mut ctx = Context::empty();
            ctx.var("pi", consts::PI);
            ctx.var("e", consts::E);

            ctx.func("sqrt", f64::sqrt);
            ctx.func("exp", f64::exp);
            ctx.func("ln", f64::ln);
            ctx.func("abs", f64::abs);
            ctx.func("sin", f64::sin);
            ctx.func("cos", f64::cos);
            ctx.func("tan", f64::tan);
            ctx.func("asin", f64::asin);
            ctx.func("acos", f64::acos);
            ctx.func("atan", f64::atan);
            ctx.func("sinh", f64::sinh);
            ctx.func("cosh", f64::cosh);
            ctx.func("tanh", f64::tanh);
            ctx.func("asinh", f64::asinh);
            ctx.func("acosh", f64::acosh);
            ctx.func("atanh", f64::atanh);
            ctx.func("floor", f64::floor);
            ctx.func("ceil", f64::ceil);
            ctx.func("round", f64::round);
            ctx.func("signum", f64::signum);
            ctx.func2("atan2", f64::atan2);
            ctx.funcn("max", max_array, 1..);
            ctx.funcn("min", min_array, 1..);
            ctx
        });

        DEFAULT_CONTEXT.with(|ctx| ctx.clone())
    }

    /// Creates an empty contexts.
    pub fn empty() -> Context<'a> {
        Context {
            vars: ContextHashMap::default(),
            funcs: ContextHashMap::default(),
        }
    }

    /// Adds a new variable/constant.
    pub fn var<S: Into<String>>(&mut self, var: S, value: f64) -> &mut Self {
        self.vars.insert(var.into(), value);
        self
    }

    /// Adds a new function of one argument.
    pub fn func<S, F>(&mut self, name: S, func: F) -> &mut Self
    where
        S: Into<String>,
        F: Fn(f64) -> f64 + 'a,
    {
        self.funcs.insert(
            name.into(),
            Rc::new(move |args: &[f64]| {
                if args.len() == 1 {
                    Ok(func(args[0]))
                } else {
                    Err(FuncEvalError::NumberArgs(1))
                }
            }),
        );
        self
    }

    /// Adds a new function of two arguments.
    pub fn func2<S, F>(&mut self, name: S, func: F) -> &mut Self
    where
        S: Into<String>,
        F: Fn(f64, f64) -> f64 + 'a,
    {
        self.funcs.insert(
            name.into(),
            Rc::new(move |args: &[f64]| {
                if args.len() == 2 {
                    Ok(func(args[0], args[1]))
                } else {
                    Err(FuncEvalError::NumberArgs(2))
                }
            }),
        );
        self
    }

    /// Adds a new function of three arguments.
    pub fn func3<S, F>(&mut self, name: S, func: F) -> &mut Self
    where
        S: Into<String>,
        F: Fn(f64, f64, f64) -> f64 + 'a,
    {
        self.funcs.insert(
            name.into(),
            Rc::new(move |args: &[f64]| {
                if args.len() == 3 {
                    Ok(func(args[0], args[1], args[2]))
                } else {
                    Err(FuncEvalError::NumberArgs(3))
                }
            }),
        );
        self
    }

    /// Adds a new function of a variable number of arguments.
    ///
    /// `n_args` specifies the allowed number of variables by giving an exact number `n` or a range
    /// `n..m`, `..`, `n..`, `..m`. The range is half-open, exclusive on the right, as is common in
    /// Rust standard library.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctx = meval::Context::empty();
    ///
    /// // require exactly 2 arguments
    /// ctx.funcn("sum_two", |xs| xs[0] + xs[1], 2);
    ///
    /// // allow an arbitrary number of arguments
    /// ctx.funcn("sum", |xs| xs.iter().sum(), ..);
    /// ```
    pub fn funcn<S, F, N>(&mut self, name: S, func: F, n_args: N) -> &mut Self
    where
        S: Into<String>,
        F: Fn(&[f64]) -> f64 + 'a,
        N: ArgGuard,
    {
        self.funcs.insert(name.into(), n_args.to_arg_guard(func));
        self
    }
}

impl<'a> Default for Context<'a> {
    fn default() -> Self {
        Context::new()
    }
}

type GuardedFunc<'a> = Rc<Fn(&[f64]) -> Result<f64, FuncEvalError> + 'a>;

/// Trait for types that can specify the number of required arguments for a function with a
/// variable number of arguments.
///
/// # Example
///
/// ```rust
/// let mut ctx = meval::Context::empty();
///
/// // require exactly 2 arguments
/// ctx.funcn("sum_two", |xs| xs[0] + xs[1], 2);
///
/// // allow an arbitrary number of arguments
/// ctx.funcn("sum", |xs| xs.iter().sum(), ..);
/// ```
pub trait ArgGuard {
    fn to_arg_guard<'a, F: Fn(&[f64]) -> f64 + 'a>(self, func: F) -> GuardedFunc<'a>;
}

impl ArgGuard for usize {
    fn to_arg_guard<'a, F: Fn(&[f64]) -> f64 + 'a>(self, func: F) -> GuardedFunc<'a> {
        Rc::new(move |args: &[f64]| {
            if args.len() == self {
                Ok(func(args))
            } else {
                Err(FuncEvalError::NumberArgs(1))
            }
        })
    }
}

impl ArgGuard for std::ops::RangeFrom<usize> {
    fn to_arg_guard<'a, F: Fn(&[f64]) -> f64 + 'a>(self, func: F) -> GuardedFunc<'a> {
        Rc::new(move |args: &[f64]| {
            if args.len() >= self.start {
                Ok(func(args))
            } else {
                Err(FuncEvalError::TooFewArguments)
            }
        })
    }
}

impl ArgGuard for std::ops::RangeTo<usize> {
    fn to_arg_guard<'a, F: Fn(&[f64]) -> f64 + 'a>(self, func: F) -> GuardedFunc<'a> {
        Rc::new(move |args: &[f64]| {
            if args.len() < self.end {
                Ok(func(args))
            } else {
                Err(FuncEvalError::TooManyArguments)
            }
        })
    }
}

impl ArgGuard for std::ops::Range<usize> {
    fn to_arg_guard<'a, F: Fn(&[f64]) -> f64 + 'a>(self, func: F) -> GuardedFunc<'a> {
        Rc::new(move |args: &[f64]| {
            if args.len() >= self.start && args.len() < self.end {
                Ok(func(args))
            } else if args.len() < self.start {
                Err(FuncEvalError::TooFewArguments)
            } else {
                Err(FuncEvalError::TooManyArguments)
            }
        })
    }
}

impl ArgGuard for std::ops::RangeFull {
    fn to_arg_guard<'a, F: Fn(&[f64]) -> f64 + 'a>(self, func: F) -> GuardedFunc<'a> {
        Rc::new(move |args: &[f64]| Ok(func(args)))
    }
}

impl<'a> ContextProvider for Context<'a> {
    fn get_var(&self, name: &str) -> Option<f64> {
        self.vars.get(name).cloned()
    }
    fn eval_func(&self, name: &str, args: &[f64]) -> Result<f64, FuncEvalError> {
        self.funcs
            .get(name)
            .map_or(Err(FuncEvalError::UnknownFunction), |f| f(args))
    }
}

#[cfg(feature = "serde")]
pub mod de {
    use super::Expr;
    use serde;
    use std::fmt;
    use std::str::FromStr;
    use tokenizer::Token;

    impl<'de> serde::Deserialize<'de> for Expr {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct ExprVisitor;

            impl<'de> serde::de::Visitor<'de> for ExprVisitor {
                type Value = Expr;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a math expression")
                }

                fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Expr::from_str(v).map_err(serde::de::Error::custom)
                }

                fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(Expr {
                        rpn: vec![Token::Number(v)],
                    })
                }

                fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(Expr {
                        rpn: vec![Token::Number(v as f64)],
                    })
                }

                fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(Expr {
                        rpn: vec![Token::Number(v as f64)],
                    })
                }
            }

            deserializer.deserialize_any(ExprVisitor)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use de::as_f64;
        use serde_json;
        use serde_test;
        #[test]
        fn test_deserialization() {
            use serde_test::Token;
            let expr = Expr::from_str("sin(x)").unwrap();

            serde_test::assert_de_tokens(&expr, &[Token::Str("sin(x)")]);
            serde_test::assert_de_tokens(&expr, &[Token::String("sin(x)")]);

            let expr = Expr::from_str("5").unwrap();

            serde_test::assert_de_tokens(&expr, &[Token::F64(5.)]);
            serde_test::assert_de_tokens(&expr, &[Token::U8(5)]);
            serde_test::assert_de_tokens(&expr, &[Token::I8(5)]);
        }

        #[test]
        fn test_json_deserialization() {
            #[derive(Deserialize)]
            struct Ode {
                #[serde(deserialize_with = "as_f64")]
                x0: f64,
                #[serde(deserialize_with = "as_f64")]
                t0: f64,
                f: Expr,
                g: Expr,
                h: Expr,
            }

            let config = r#"
            {
                "x0": "cos(1.)",
                "t0": 2,
                "f": "sin(x)",
                "g": 2.5,
                "h": 5
            }
            "#;
            let ode: Ode = serde_json::from_str(config).unwrap();

            assert_eq!(ode.x0, 1f64.cos());
            assert_eq!(ode.t0, 2f64);
            assert_eq!(ode.f.bind("x").unwrap()(2.), 2f64.sin());
            assert_eq!(ode.g.eval().unwrap(), 2.5f64);
            assert_eq!(ode.h.eval().unwrap(), 5f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use Error;

    #[test]
    fn test_eval() {
        assert_eq!(eval_str("2 + 3"), Ok(5.));
        assert_eq!(eval_str("2 + (3 + 4)"), Ok(9.));
        assert_eq!(eval_str("-2^(4 - 3) * (3 + 4)"), Ok(-14.));
        assert_eq!(eval_str("a + 3"), Err(Error::UnknownVariable("a".into())));
        assert_eq!(eval_str("round(sin (pi) * cos(0))"), Ok(0.));
        assert_eq!(eval_str("round( sqrt(3^2 + 4^2)) "), Ok(5.));
        assert_eq!(eval_str("max(1.)"), Ok(1.));
        assert_eq!(eval_str("max(1., 2., -1)"), Ok(2.));
        assert_eq!(eval_str("min(1., 2., -1)"), Ok(-1.));
        assert_eq!(
            eval_str("sin(1.) + cos(2.)"),
            Ok((1f64).sin() + (2f64).cos())
        );
        assert_eq!(eval_str("10 % 9"), Ok(10f64 % 9f64));
    }

    #[test]
    fn test_builtins() {
        assert_eq!(eval_str("atan2(1.,2.)"), Ok((1f64).atan2(2.)));
    }

    #[test]
    fn test_eval_func_ctx() {
        use std::collections::{BTreeMap, HashMap};
        let y = 5.;
        assert_eq!(
            eval_str_with_context("phi(2.)", Context::new().func("phi", |x| x + y + 3.)),
            Ok(2. + y + 3.)
        );
        assert_eq!(
            eval_str_with_context(
                "phi(2., 3.)",
                Context::new().func2("phi", |x, y| x + y + 3.)
            ),
            Ok(2. + 3. + 3.)
        );
        assert_eq!(
            eval_str_with_context(
                "phi(2., 3., 4.)",
                Context::new().func3("phi", |x, y, z| x + y * z)
            ),
            Ok(2. + 3. * 4.)
        );
        assert_eq!(
            eval_str_with_context(
                "phi(2., 3.)",
                Context::new().funcn("phi", |xs: &[f64]| xs[0] + xs[1], 2)
            ),
            Ok(2. + 3.)
        );
        let mut m = HashMap::new();
        m.insert("x", 2.);
        m.insert("y", 3.);
        assert_eq!(eval_str_with_context("x + y", &m), Ok(2. + 3.));
        assert_eq!(
            eval_str_with_context("x + z", m),
            Err(Error::UnknownVariable("z".into()))
        );
        let mut m = BTreeMap::new();
        m.insert("x", 2.);
        m.insert("y", 3.);
        assert_eq!(eval_str_with_context("x + y", &m), Ok(2. + 3.));
        assert_eq!(
            eval_str_with_context("x + z", m),
            Err(Error::UnknownVariable("z".into()))
        );
    }

    #[test]
    fn test_bind() {
        let expr = Expr::from_str("x + 3").unwrap();
        let func = expr.clone().bind("x").unwrap();
        assert_eq!(func(1.), 4.);

        assert_eq!(
            expr.clone().bind("y").err(),
            Some(Error::UnknownVariable("x".into()))
        );

        let ctx = (("x", 2.), builtin());
        let func = expr.bind_with_context(&ctx, "y").unwrap();
        assert_eq!(func(1.), 5.);

        let expr = Expr::from_str("x + y + 2.").unwrap();
        let func = expr.clone().bind2("x", "y").unwrap();
        assert_eq!(func(1., 2.), 5.);
        assert_eq!(
            expr.clone().bind2("z", "y").err(),
            Some(Error::UnknownVariable("x".into()))
        );
        assert_eq!(
            expr.bind2("x", "z").err(),
            Some(Error::UnknownVariable("y".into()))
        );

        let expr = Expr::from_str("x + y^2 + z^3").unwrap();
        let func = expr.clone().bind3("x", "y", "z").unwrap();
        assert_eq!(func(1., 2., 3.), 32.);

        let expr = Expr::from_str("sin(x)").unwrap();
        let func = expr.clone().bind("x").unwrap();
        assert_eq!(func(1.), (1f64).sin());

        let expr = Expr::from_str("sin(x,2)").unwrap();
        match expr.clone().bind("x") {
            Err(Error::Function(_, FuncEvalError::NumberArgs(1))) => {}
            _ => panic!("bind did not error"),
        }
        let expr = Expr::from_str("hey(x,2)").unwrap();
        match expr.clone().bind("x") {
            Err(Error::Function(_, FuncEvalError::UnknownFunction)) => {}
            _ => panic!("bind did not error"),
        }
    }

    #[test]
    fn hash_context() {
        let y = 0.;
        {
            let z = 0.;

            let mut ctx = Context::new();
            ctx.var("x", 1.).func("f", |x| x + y).func("g", |x| x + z);
            ctx.func2("g", |x, y| x + y);
        }
    }
}
