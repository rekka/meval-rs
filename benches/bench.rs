#![feature(test)]

extern crate meval;
extern crate test;

use meval::max_array;
use meval::min_array;
use meval::Context;
use meval::ContextProvider;
use meval::Expr;
use meval::FuncEvalError;
use std::f64::consts;
use test::Bencher;

const EXPR: &str = "abs(sin(x + 1) * (x^2 + x + 1))";

#[bench]
fn parsing(b: &mut Bencher) {
    b.iter(|| {
        EXPR.parse::<Expr>().unwrap();
    });
}

#[bench]
fn evaluation_matchcontext(b: &mut Bencher) {
    let expr: Expr = EXPR.parse().unwrap();
    let func = expr.bind_with_context(MatchBuiltins, "x").unwrap();
    b.iter(|| {
        func(1.);
    });
}

#[bench]
fn evaluation_hashcontext(b: &mut Bencher) {
    let expr: Expr = EXPR.parse().unwrap();
    let func = expr.bind_with_context(Context::new(), "x").unwrap();
    b.iter(|| {
        func(1.);
    });
}

#[bench]
fn default_context(b: &mut Bencher) {
    let expr: Expr = "1 + 2 * 3".parse().unwrap();
    b.iter(|| expr.eval());
}

macro_rules! one_arg {
    ($args:expr, $func:ident) => {
        if $args.len() == 1 {
            Ok($args[0].$func())
        } else {
            Err(FuncEvalError::NumberArgs(1))
        }
    };
}

macro_rules! two_args {
    ($args:expr, $func:ident) => {
        if $args.len() == 2 {
            Ok($args[0].$func($args[1]))
        } else {
            Err(FuncEvalError::NumberArgs(2))
        }
    };
}

macro_rules! one_or_more_arg {
    ($args:expr, $func:ident) => {
        if $args.len() >= 1 {
            Ok($func($args))
        } else {
            Err(FuncEvalError::TooFewArguments)
        }
    };
}

/// Built-in functions and constants.
///
/// See the library documentation for the list of built-ins.
#[doc(hidden)]
pub struct MatchBuiltins;

impl ContextProvider for MatchBuiltins {
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
